# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
import tempfile
import tomllib

from click.testing import CliRunner

from synalinks.cli.main import cli
from synalinks.src import testing


class InitTest(testing.TestCase):
    def _init(self, name, template, description):
        with tempfile.TemporaryDirectory() as d:
            cwd = os.getcwd()
            os.chdir(d)
            try:
                result = CliRunner().invoke(
                    cli, ["init", "-n", name, "-t", template, "-d", description]
                )
                self.assertEqual(result.exit_code, 0, result.output)
                path = os.path.join(name, "pyproject.toml")
                if not os.path.exists(path):
                    path = os.path.join(name, "backend", "pyproject.toml")
                with open(path, "rb") as f:
                    return tomllib.load(f)["project"]
            finally:
                os.chdir(cwd)

    def test_quotes_in_description_keep_pyproject_valid(self):
        project = self._init("qa", "script", 'My "QA" bot\\')
        self.assertEqual(project["name"], "qa")
        self.assertEqual(project["description"], 'My "QA" bot\\')

    def test_app_template_rewrites_backend_pyproject(self):
        project = self._init("my-app", "app", "My app")
        self.assertEqual(project["name"], "my-app")
        self.assertEqual(project["description"], "My app")
