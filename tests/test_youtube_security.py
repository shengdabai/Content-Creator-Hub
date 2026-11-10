import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


class VideoSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # No API client or local settings are loaded during tests.
        modules = {'anthropic': types.SimpleNamespace(Anthropic=Mock()),
                   'dotenv': types.SimpleNamespace(load_dotenv=Mock())}
        source = Path(os.environ.get('VIDEO_AGENT_SOURCE', ROOT / 'youtube-creator/youtube-creator.py'))
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        spec = importlib.util.spec_from_file_location('video_agent', source)
        self.agent = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules), patch('pathlib.Path.cwd', return_value=self.root):
            spec.loader.exec_module(self.agent)

    def test_shell_tool_is_unavailable(self):
        self.assertNotIn('bash', [tool['name'] for tool in self.agent.TOOLS])
        with patch.object(self.agent.subprocess, 'run') as run:
            self.assertIn('disabled', self.agent.run_bash('echo unsafe'))
            self.assertIn('Unknown tool', self.agent.execute_tool('bash', {'command': 'echo unsafe'}))
            run.assert_not_called()

    def test_file_tools_reject_escape_and_links(self):
        outside = self.root / 'outside.txt'
        outside.write_text('private fixture')
        (self.root / 'output/link').symlink_to(self.root)
        os.link(outside, self.root / 'output/hardlink.txt')
        for path in ('outside.txt', str(outside), 'output/../outside.txt',
                     'output/link/outside.txt', 'output/hardlink.txt', 'output/.hidden'):
            with self.subTest(path=path):
                self.assertTrue(self.agent.run_read_file(path).startswith('Error:'))
                self.assertTrue(self.agent.run_write_file(path, 'overwrite').startswith('Error:'))
        self.assertEqual(outside.read_text(), 'private fixture')
        self.assertIn('已写入', self.agent.run_write_file('output/script.json', '{}'))
        self.assertEqual(self.agent.run_read_file('output/script.json'), '{}')

    def test_tts_text_and_filename_are_literal_arguments(self):
        payload = '"; $(touch injected); `whoami`'
        self.agent.run_write_file('output/script.json', json.dumps({'segments': [{'text': payload}]}))
        audio = self.root / 'output/audio'
        audio.mkdir()
        (audio / 'seg_999.mp3').write_bytes(b'stale')
        with patch.object(self.agent.subprocess, 'run', return_value=Mock(returncode=0, stdout='2.5', stderr='')) as run:
            result = self.agent.execute_tool('generate_assets', {'script_path': 'output/script.json'})
        self.assertIn('素材生成完成', result)
        for call in run.call_args_list:
            self.assertIsInstance(call.args[0], list)
            self.assertFalse(call.kwargs.get('shell', False))
        tts = run.call_args_list[1].args[0]
        self.assertEqual(tts[tts.index('--text') + 1], payload)
        self.assertEqual((self.root / 'output/audio_list.txt').read_text(), "file 'audio/seg_000.mp3'")

    def test_failed_tts_stops_before_stale_audio_merge(self):
        self.agent.run_write_file('output/script.json', '{"segments":[{"text":"hello"}]}')
        responses = [Mock(returncode=0), Mock(returncode=1, stderr='failed')]
        with patch.object(self.agent.subprocess, 'run', side_effect=responses) as run:
            result = self.agent.execute_tool('generate_assets', {'script_path': 'output/script.json'})
        self.assertTrue(result.startswith('Error:'))
        self.assertEqual(run.call_count, 2)

    def test_compose_uses_argument_list_and_rejects_outside_output(self):
        (self.root / 'output/narration.mp3').write_bytes(b'audio')
        out = self.root / 'output/final.mp4'
        out.write_bytes(b'video')
        with patch.object(self.agent.subprocess, 'run', return_value=Mock(returncode=0, stdout='3', stderr='')) as run:
            self.assertIn('视频合成完成', self.agent.execute_tool('compose_video', {'output_path': 'output/final.mp4'}))
            self.assertIsInstance(run.call_args.args[0], list)
            self.assertFalse(run.call_args.kwargs.get('shell', False))
            run.reset_mock()
            self.assertTrue(self.agent.execute_tool('compose_video', {'output_path': '../outside.mp4'}).startswith('Error:'))
            run.assert_not_called()

    def test_upload_requires_human_approval_before_credentials(self):
        (self.root / 'output/final.mp4').write_bytes(b'video')
        args = {'video_path': 'output/final.mp4', 'title': 'title', 'description': 'description'}
        with patch.object(sys.stdin, 'isatty', return_value=False):
            self.assertIn('interactive terminal', self.agent.execute_tool('upload_youtube', args))
        with patch.object(sys.stdin, 'isatty', return_value=True), patch.object(sys.stdout, 'isatty', return_value=True), patch('builtins.input', return_value='no'):
            self.assertEqual(self.agent.execute_tool('upload_youtube', args), 'Upload cancelled')

    def test_missing_binary_returns_tool_error(self):
        self.agent.run_write_file('output/script.json', '{"segments":[{"text":"hello"}]}')
        with patch.object(self.agent.subprocess, 'run', side_effect=FileNotFoundError):
            self.assertTrue(self.agent.execute_tool('generate_assets', {'script_path': 'output/script.json'}).startswith('Error:'))


if __name__ == '__main__':
    unittest.main()
