import logging
from pathlib import Path

import pytest
from click.testing import CliRunner

from ragamuffin.cli.muffin import cli
from ragamuffin.storage.utils import get_storage
from tests.utils import env_vars


@pytest.mark.parametrize("storage_type", ["file", "cassandra"])
def test_muffin_cli_from_files(storage_type):
    with env_vars(
        RAGAMUFFIN_STORAGE_TYPE=storage_type,
        RAGAMUFFIN_EMBEDDING_DIMENSION="312",
        RAGAMUFFIN_EMBEDDING_MODEL="huggingface.co/huawei-noah/TinyBERT_General_4L_312D",
    ):
        runner = CliRunner()

        agent_name = "test_agent"
        test_data_path = Path(__file__).parent / "data" / "udhr"

        result = runner.invoke(cli, ["generate", "from_files", agent_name, str(test_data_path)])
        assert result.exit_code == 0
        assert agent_name in get_storage().list_agents()

        result = runner.invoke(cli, ["delete", agent_name])
        assert result.exit_code == 0
        assert agent_name not in get_storage().list_agents()


@pytest.mark.parametrize("storage_type", ["file", "cassandra"])
def test_muffin_cli_from_git(caplog, storage_type):
    with env_vars(
        RAGAMUFFIN_STORAGE_TYPE=storage_type,
        RAGAMUFFIN_EMBEDDING_DIMENSION="312",
        RAGAMUFFIN_EMBEDDING_MODEL="huggingface.co/huawei-noah/TinyBERT_General_4L_312D",
    ):
        runner = CliRunner()
        caplog.set_level(logging.INFO, logger="ragamuffin")
        agent_name = "test_agent"

        result = runner.invoke(cli, ["generate", "from_git", agent_name, "https://github.com/octocat/Hello-World/"])
        assert result.exit_code == 0
        assert agent_name in get_storage().list_agents()

        caplog.clear()
        result = runner.invoke(cli, ["agents"])
        assert result.exit_code == 0
        assert agent_name in caplog.text

        result = runner.invoke(cli, ["delete", agent_name])
        assert result.exit_code == 0
        assert agent_name not in get_storage().list_agents()
