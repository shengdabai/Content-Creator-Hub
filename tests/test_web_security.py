import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'x-content-mvp'))
from app.services import ingest
with patch('dotenv.load_dotenv'):
    from app import config, db
    config.OPENAI_API_KEY = ''
    from app.main import app
from fastapi.testclient import TestClient


class WebSecurityTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        stack.enter_context(patch.object(db, 'DATA_DIR', directory))
        stack.enter_context(patch.object(db, 'DB_PATH', directory / 'test.db'))

    def test_cross_site_forms_and_api_writes_are_rejected(self):
        with TestClient(app, base_url='http://localhost:8000') as client:
            for path in ('/sources/create', '/style/import', '/api/sources'):
                for origin in ('https://attacker.example', 'null'):
                    response = client.post(path, headers={'Origin': origin})
                    self.assertEqual(response.status_code, 403)
            response = client.post('/api/sources', json={}, headers={'Origin': 'http://127.0.0.1:8000'})
            self.assertEqual(response.status_code, 400)

    def test_untrusted_host_is_rejected(self):
        with TestClient(app, base_url='http://attacker.example') as client:
            self.assertEqual(client.get('/api/drafts').status_code, 400)

    def test_nonpublic_urls_and_mixed_dns_are_blocked(self):
        for address in ('127.0.0.1', '10.0.0.1', '169.254.169.254', '::1', '100.64.0.1'):
            with self.subTest(address=address), patch.object(ingest.socket, 'getaddrinfo', return_value=[(None, None, None, None, (address, 0))]):
                with self.assertRaises(ValueError):
                    ingest.validate_public_http_url('https://example.com')
        answers = [(None, None, None, None, (ip, 0)) for ip in ('8.8.8.8', '10.0.0.1')]
        with patch.object(ingest.socket, 'getaddrinfo', return_value=answers), self.assertRaises(ValueError):
            ingest.validate_public_http_url('https://example.com')

    def test_checked_ip_is_used_with_original_tls_hostname(self):
        response = Mock(status=200)
        response.read.return_value = b'<title>Example</title><article>Content</article>'
        pool = Mock()
        pool.urlopen.return_value = response
        answers = [(None, None, None, None, ('8.8.8.8', 0))]
        with patch.object(ingest.socket, 'getaddrinfo', return_value=answers) as dns, patch.object(ingest.urllib3, 'HTTPSConnectionPool', return_value=pool) as factory:
            self.assertEqual(ingest.extract_web_content('https://example.com:8443/a?q=1')['text'], 'Content')
        self.assertEqual(dns.call_count, 1)
        self.assertEqual(factory.call_args.args, ('8.8.8.8', 8443))
        self.assertEqual(factory.call_args.kwargs['server_hostname'], 'example.com')
        self.assertEqual(factory.call_args.kwargs['assert_hostname'], 'example.com')
        self.assertEqual(pool.urlopen.call_args.args, ('GET', '/a?q=1'))
        self.assertFalse(pool.urlopen.call_args.kwargs['redirect'])
        self.assertFalse(pool.urlopen.call_args.kwargs['retries'])
        response.close.assert_called_once()
        pool.close.assert_called_once()

    def test_redirects_and_large_pages_fail_and_close_connections(self):
        for status, body in ((302, b''), (200, b'x' * (ingest.MAX_PAGE_BYTES + 1))):
            response = Mock(status=status)
            response.read.return_value = body
            pool = Mock()
            pool.urlopen.return_value = response
            with self.subTest(status=status), patch.object(ingest, '_validated_target', return_value=(ingest.urlparse('http://example.com'), 80, '8.8.8.8')), patch.object(ingest.urllib3, 'HTTPConnectionPool', return_value=pool):
                with self.assertRaises(ValueError):
                    ingest.extract_web_content('http://example.com')
            response.close.assert_called_once()
            pool.close.assert_called_once()

    def test_rescoring_preserves_engagement(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(db, 'DATA_DIR', Path(directory)), patch.object(db, 'DB_PATH', Path(directory) / 'test.db'):
            with TestClient(app, base_url='http://localhost:8000') as client:
                for path in ('/', '/style', '/drafts', '/healthz'):
                    self.assertEqual(client.get(path).status_code, 200)
                created = client.post('/api/sources', json={'text_input': '<script>alert(1)</script>A concrete lesson.'})
                self.assertEqual(created.status_code, 200)
                source_id = created.json()['source']['id']
                page = client.get(f'/sources/{source_id}')
                self.assertEqual(page.status_code, 200)
                self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;', page.text)
                before = {d['id']: d['engagement'] for d in db.list_source_drafts(source_id)}
                self.assertTrue(any(value > 0 for value in before.values()))
                self.assertEqual(client.post(f'/api/drafts/score/{source_id}').status_code, 200)
                self.assertEqual(before, {d['id']: d['engagement'] for d in db.list_source_drafts(source_id)})


if __name__ == '__main__':
    unittest.main()
