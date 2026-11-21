import unittest
from unittest.mock import patch, Mock
from verifier import ClaimVerifier


def source(url='https://example.com/news'):
    return dict(url=url, title='Article', source_domain='example.com', publisher='Example',
                published_at='2026-11-21', snippet='Preview', verdict='not_analyzed', reason='Not assessed')


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.v = ClaimVerifier()

    def test_conflicting_sources_remain_mixed_and_unread_links_survive(self):
        sources = [source(), source('https://example.org/news'), source('https://example.net/news')]
        analysis = [{**sources[0], 'verdict': 'supports', 'reason': 'Supports'},
                    {**sources[1], 'verdict': 'contradicts', 'reason': 'Contradicts'}]
        r = self.v._aggregate_results('Claim', analysis, sources)
        self.assertEqual(r['verdict'], 'Mixed evidence')
        self.assertEqual(len(r['sources']), 3)
        self.assertNotIn('confidence', r)

    def test_no_evidence_does_not_mean_false(self):
        self.assertEqual(self.v._aggregate_results('Claim', [], [])['verdict'], 'Insufficient evidence')

    def test_context_and_quote_required_for_contradiction(self):
        article = {**source(), 'content': 'A different match ended with 74 runs.'}
        for context, quote in [(False, article['content']), (True, 'Fabricated quote')]:
            with patch.object(self.v, '_llm', return_value=dict(verdict='CONTRADICTS', reason='Wrong match', same_context=context, quote=quote)):
                r = self.v._analyze_with_llm('A century', [article], [])
                self.assertEqual(r[0]['verdict'], 'insufficient')

    def test_valid_support_preserves_excerpt(self):
        article = {**source(), 'content': 'The launch happened on Monday.'}
        with patch.object(self.v, '_llm', return_value=dict(verdict='SUPPORTS', reason='Matching launch', same_context=True, quote=article['content'])):
            r = self.v._analyze_with_llm('Launch on Monday', [article], [])
            self.assertEqual(r[0]['verdict'], 'supports')
            self.assertEqual(r[0]['quote'], article['content'])

    def test_ambiguous_claim_stops_before_search(self):
        with patch.object(self.v, '_llm', return_value=None), patch.object(self.v, '_search_sources') as search:
            r = self.v.verify('Virat hit a century')
            self.assertEqual(r['verdict'], 'Needs clarification')
            search.assert_not_called()

    def test_claim_rephrased_as_yes_no_question_must_search(self):
        plan = dict(query='Tamil Nadu CM Vijay', needs_clarification=True,
                    missing_context='entity', question='Is Vijay currently the Chief Minister of Tamil Nadu?')
        with patch.object(self.v, '_llm', return_value=plan), patch.object(self.v, '_search_sources', return_value=[]) as search:
            r = self.v.verify('CM of tamilnadu is Vijay')
            search.assert_called_once()
            self.assertNotEqual(r['verdict'], 'Needs clarification')

    def test_which_vijay_response_cannot_block_search(self):
        for question in ['Which Vijay is being referred to as the Chief Minister of Tamil Nadu?',
                         'Which date?', 'Where is Tamil Nadu?']:
            plan = dict(query='Chief Minister Tamil Nadu Vijay', needs_clarification=True,
                        missing_context='entity', question=question)
            with self.subTest(question=question), patch.object(self.v, '_llm', return_value=plan), patch.object(self.v, '_search_sources', return_value=[]) as search:
                r = self.v.verify('CM of tamilnadu is Vijay')
                search.assert_called_once_with('Chief Minister Tamil Nadu Vijay', unittest.mock.ANY)
                self.assertNotEqual(r['verdict'], 'Needs clarification')

    def test_current_office_does_not_need_date(self):
        plan = dict(query='Tamil Nadu CM Vijay', needs_clarification=True,
                    missing_context='date', question='Which date do you mean?')
        with patch.object(self.v, '_llm', return_value=plan):
            self.assertIsNone(self.v._prepare_claim('CM of tamilnadu is Vijay', [])[1])

    def test_genuine_missing_match_still_requests_context(self):
        plan = dict(query='Virat century', needs_clarification=True,
                    missing_context='event', question='Which match do you mean?')
        with patch.object(self.v, '_llm', return_value=plan):
            self.assertEqual(self.v.verify('Virat hit a century')['verdict'], 'Needs clarification')

    def test_query_fallback_preserves_entities_and_date(self):
        with patch.object(self.v, '_llm', return_value=None):
            q, question = self.v._prepare_claim('Vijay is the CM of tamilnadu in 2026', [])
            self.assertIn('Vijay', q)
            self.assertIn('Chief Minister', q)
            self.assertIn('Tamil Nadu', q)
            self.assertIn('2026', q)
            self.assertIsNone(question)

    def test_provider_credit_error_visible_and_other_provider_survives(self):
        good = Mock(ok=True)
        good.json.return_value = {'articles': [dict(url='https://example.com/news', title='Article', source={'name': 'Example'})]}
        bad = Mock(ok=False, status_code=432, text='Usage limit exceeded')
        with patch.dict('os.environ', {'GNEWS_API_KEY': 'test', 'TAVILY_API_KEY': 'test'}), patch('verifier.requests.get', return_value=good), patch('verifier.requests.post', return_value=bad):
            notices = []
            rows = self.v._search_sources('query', notices)
        self.assertEqual(len(rows), 1)
        self.assertIn('credits', notices[-1]['message'])

    def test_tavily_mapping_basic_cost_and_deduplication(self):
        news = Mock(ok=True)
        news.json.return_value = {'articles': [dict(url='https://example.com/news', title='News', source={'name': 'Example'})]}
        web = Mock(ok=True)
        web.json.return_value = {'results': [
            dict(url='https://example.com/news', title='Duplicate', content='Duplicate preview'),
            dict(url='https://example.org/article', title='Web evidence', content='Relevant preview', published_date='2026-11-21'),
            dict(url='javascript:alert(1)', title='Unsafe')]}
        with patch.dict('os.environ', {'GNEWS_API_KEY': 'test', 'TAVILY_API_KEY': 'test'}), patch('verifier.requests.get', return_value=news), patch('verifier.requests.post', return_value=web) as post:
            notices = []
            rows = self.v._search_sources('query', notices)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]['provider'], 'Tavily')
        self.assertEqual(rows[1]['snippet'], 'Relevant preview')
        self.assertEqual(rows[1]['published_at'], '2026-11-21')
        self.assertEqual(post.call_args.args[0], 'https://api.tavily.com/search')
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['search_depth'], 'basic')
        self.assertFalse(payload['auto_parameters'])
        self.assertFalse(payload['include_answer'])
        self.assertNotIn('api_key', payload)
        self.assertEqual(notices[-1]['provider'], 'Tavily')

    def test_missing_tavily_key_does_not_call_serper(self):
        news = Mock(ok=True)
        news.json.return_value = {'articles': []}
        with patch.dict('os.environ', {'GNEWS_API_KEY': 'test', 'TAVILY_API_KEY': '', 'SERPER_API_KEY': 'unused'}), patch('verifier.requests.get', return_value=news), patch('verifier.requests.post') as post:
            notices = []
            self.v._search_sources('query', notices)
            post.assert_not_called()
        self.assertEqual(notices[-1]['provider'], 'Tavily')
        self.assertEqual(notices[-1]['status'], 'unavailable')

    def test_invalid_llm_json_does_not_drop_links(self):
        self.v.groq_key = 'test'
        response = Mock(ok=True)
        response.json.return_value = {'choices': [{'message': {'content': 'not json'}}]}
        notices = []
        with patch('verifier.requests.post', return_value=response):
            self.assertIsNone(self.v._llm('instructions', {}, notices))
        self.assertEqual(notices[0]['status'], 'error')

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            self.v.verify(' ')

    def test_direct_answer_uses_only_existing_evidence_links(self):
        row = {**source(), 'verdict': 'contradicts', 'quote': 'The launch was Tuesday.'}
        result = self.v._aggregate_results('Launch was Monday', [row], [row])
        with patch.object(self.v, '_llm', return_value={'answer': 'The cited article says Tuesday, not Monday.', 'source_ids': [1]}):
            answer = self.v._compose_answer('Launch was Monday', result, [])
        self.assertEqual(answer['status'], 'generated')
        self.assertEqual(answer['citations'][0]['url'], row['url'])

    def test_answer_rejects_fabricated_citation(self):
        row = {**source(), 'verdict': 'supports', 'quote': 'Evidence.'}
        result = self.v._aggregate_results('Claim', [row], [row])
        for ids in [[99], ['1'], [True], []]:
            with self.subTest(ids=ids), patch.object(self.v, '_llm', return_value={'answer': 'Answer', 'source_ids': ids}):
                self.assertEqual(self.v._compose_answer('Claim', result, [])['status'], 'summary')

    def test_mixed_answer_must_cite_both_sides(self):
        rows = [{**source(), 'verdict': 'supports', 'quote': 'Supports.'},
                {**source('https://example.org/news'), 'verdict': 'contradicts', 'quote': 'Contradicts.'}]
        result = self.v._aggregate_results('Claim', rows, rows)
        with patch.object(self.v, '_llm', return_value={'answer': 'Yes', 'source_ids': [1]}):
            self.assertEqual(self.v._compose_answer('Claim', result, [])['status'], 'summary')
        with patch.object(self.v, '_llm', return_value={'answer': 'The sources disagree.', 'source_ids': [1, 2]}):
            self.assertEqual(self.v._compose_answer('Claim', result, [])['status'], 'generated')

    def test_no_evidence_does_not_generate_answer(self):
        result = self.v._aggregate_results('Claim', [], [])
        with patch.object(self.v, '_llm') as llm:
            self.assertEqual(self.v._compose_answer('Claim', result, [])['status'], 'summary')
            llm.assert_not_called()

    def test_answer_api_failure_preserves_evidence_summary(self):
        row = {**source(), 'verdict': 'supports', 'quote': 'Evidence.'}
        result = self.v._aggregate_results('Claim', [row], [row])
        with patch.object(self.v, '_llm', return_value=None):
            answer = self.v._compose_answer('Claim', result, [])
        self.assertEqual(answer['text'], result['reasoning'])
        self.assertEqual(len(result['sources']), 1)

    def test_unsafe_links_rejected(self):
        for url in ['javascript:alert(1)', 'file:///etc/passwd', 'https://user:password@example.com']:
            self.assertIsNone(self.v._url(url))


if __name__ == '__main__':
    unittest.main()
