"""Tests for interactive setup module."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from raggy.setup.interactive import InteractiveSetup, run_interactive_setup


class TestInteractiveSetup:
    """Test cases for InteractiveSetup class."""

    def test_init_quiet_mode(self):
        """Test initialization in quiet mode."""
        setup = InteractiveSetup(quiet=True)
        assert setup.quiet is True
        assert setup.config == {}

    def test_init_interactive_mode(self):
        """Test initialization in interactive mode."""
        setup = InteractiveSetup(quiet=False)
        assert setup.quiet is False
        assert setup.config == {}

    def test_select_vector_store_quiet_returns_chromadb(self):
        """Test vector store selection in quiet mode returns default."""
        setup = InteractiveSetup(quiet=True)
        result = setup._select_vector_store()
        assert result == "chromadb"

    @patch('builtins.input', return_value='')
    def test_select_vector_store_default(self, mock_input):
        """Test vector store selection with default (Enter key)."""
        setup = InteractiveSetup(quiet=False)
        result = setup._select_vector_store()
        assert result == "chromadb"

    @patch('builtins.input', return_value='2')
    def test_select_vector_store_pinecone(self, mock_input):
        """Test vector store selection of Pinecone."""
        setup = InteractiveSetup(quiet=False)
        result = setup._select_vector_store()
        assert result == "pinecone"

    @patch('builtins.input', return_value='3')
    def test_select_vector_store_supabase(self, mock_input):
        """Test vector store selection of Supabase."""
        setup = InteractiveSetup(quiet=False)
        result = setup._select_vector_store()
        assert result == "supabase"

    @patch('builtins.input', return_value='99')
    def test_select_vector_store_invalid(self, mock_input):
        """Test vector store selection with invalid choice falls back to default."""
        setup = InteractiveSetup(quiet=False)
        result = setup._select_vector_store()
        assert result == "chromadb"

    def test_select_embedding_provider_quiet_returns_st(self):
        """Test embedding provider selection in quiet mode returns default."""
        setup = InteractiveSetup(quiet=True)
        result = setup._select_embedding_provider()
        assert result == "sentence-transformers"

    @patch('builtins.input', return_value='')
    def test_select_embedding_provider_default(self, mock_input):
        """Test embedding provider selection with default."""
        setup = InteractiveSetup(quiet=False)
        result = setup._select_embedding_provider()
        assert result == "sentence-transformers"

    @patch('builtins.input', return_value='2')
    def test_select_embedding_provider_openai(self, mock_input):
        """Test embedding provider selection of OpenAI."""
        setup = InteractiveSetup(quiet=False)
        result = setup._select_embedding_provider()
        assert result == "openai"

    def test_select_st_model_quiet_returns_default(self):
        """Test sentence-transformers model selection in quiet mode."""
        setup = InteractiveSetup(quiet=True)
        result = setup._select_st_model()
        assert result["model"] == "all-MiniLM-L6-v2"
        assert result["dimension"] == 384

    @patch('builtins.input', return_value='2')
    def test_select_st_model_accurate(self, mock_input):
        """Test sentence-transformers model selection of accurate model."""
        setup = InteractiveSetup(quiet=False)
        result = setup._select_st_model()
        assert result["model"] == "all-mpnet-base-v2"
        assert result["dimension"] == 768

    def test_select_openai_model_quiet_returns_default(self):
        """Test OpenAI model selection in quiet mode."""
        setup = InteractiveSetup(quiet=True)
        result = setup._select_openai_model()
        assert result["model"] == "text-embedding-3-small"
        assert result["dimension"] == 1536

    @patch('builtins.input', return_value='2')
    def test_select_openai_model_large(self, mock_input):
        """Test OpenAI model selection of large model."""
        setup = InteractiveSetup(quiet=False)
        result = setup._select_openai_model()
        assert result["model"] == "text-embedding-3-large"
        assert result["dimension"] == 3072

    def test_collect_vector_store_config_chromadb(self):
        """Test ChromaDB vector store configuration."""
        setup = InteractiveSetup(quiet=False)
        result = setup._collect_vector_store_config("chromadb")
        assert result == {"path": "./vectordb"}

    def test_collect_pinecone_config_quiet(self):
        """Test Pinecone configuration in quiet mode."""
        setup = InteractiveSetup(quiet=True)
        result = setup._collect_pinecone_config()
        assert result["apiKey"] == "${PINECONE_API_KEY}"
        assert result["environment"] == "us-east-1-aws"
        assert result["indexName"] == "raggy-index"

    @patch('builtins.input', side_effect=['${PINECONE_API_KEY}', 'us-west-2', 'my-index'])
    def test_collect_pinecone_config_interactive(self, mock_input):
        """Test Pinecone configuration with user inputs."""
        setup = InteractiveSetup(quiet=False)
        result = setup._collect_pinecone_config()
        assert result["apiKey"] == "${PINECONE_API_KEY}"
        assert result["environment"] == "us-west-2"
        assert result["indexName"] == "my-index"

    def test_collect_supabase_config_quiet(self):
        """Test Supabase configuration in quiet mode."""
        setup = InteractiveSetup(quiet=True)
        result = setup._collect_supabase_config()
        assert result["url"] == "${SUPABASE_URL}"
        assert result["apiKey"] == "${SUPABASE_ANON_KEY}"

    @patch('builtins.input', side_effect=['${SUPABASE_URL}', '${SUPABASE_ANON_KEY}'])
    def test_collect_supabase_config_interactive(self, mock_input):
        """Test Supabase configuration with user inputs."""
        setup = InteractiveSetup(quiet=False)
        result = setup._collect_supabase_config()
        assert result["url"] == "${SUPABASE_URL}"
        assert result["apiKey"] == "${SUPABASE_ANON_KEY}"

    def test_collect_embedding_config_sentence_transformers(self):
        """Test sentence-transformers embedding configuration."""
        setup = InteractiveSetup(quiet=False)
        model_info = {
            "model": "all-MiniLM-L6-v2",
            "dimension": 384
        }
        result = setup._collect_embedding_config("sentence-transformers", model_info)
        assert result["model"] == "all-MiniLM-L6-v2"
        assert result["device"] == "cpu"

    def test_collect_openai_config_quiet(self):
        """Test OpenAI configuration in quiet mode."""
        setup = InteractiveSetup(quiet=True)
        model_info = {"model": "text-embedding-3-small", "dimension": 1536}
        result = setup._collect_openai_config(model_info)
        assert result["apiKey"] == "${OPENAI_API_KEY}"
        assert result["model"] == "text-embedding-3-small"

    @patch('builtins.input', return_value='${OPENAI_API_KEY}')
    def test_collect_openai_config_interactive(self, mock_input):
        """Test OpenAI configuration with user input."""
        setup = InteractiveSetup(quiet=False)
        model_info = {"model": "text-embedding-3-small", "dimension": 1536}
        result = setup._collect_openai_config(model_info)
        assert result["apiKey"] == "${OPENAI_API_KEY}"
        assert result["model"] == "text-embedding-3-small"

    def test_get_embedding_dimension_sentence_transformers(self):
        """Test getting embedding dimension for sentence-transformers."""
        setup = InteractiveSetup(quiet=False)
        config = {"model": "all-MiniLM-L6-v2"}
        dimension = setup._get_embedding_dimension("sentence-transformers", config)
        assert dimension == 384

    def test_get_embedding_dimension_openai(self):
        """Test getting embedding dimension for OpenAI."""
        setup = InteractiveSetup(quiet=False)
        config = {"model": "text-embedding-3-large"}
        dimension = setup._get_embedding_dimension("openai", config)
        assert dimension == 3072

    def test_get_embedding_dimension_fallback(self):
        """Test getting embedding dimension with unknown model."""
        setup = InteractiveSetup(quiet=False)
        config = {"model": "unknown-model"}
        dimension = setup._get_embedding_dimension("sentence-transformers", config)
        assert dimension == 384  # Default fallback

    def test_build_config_chromadb_st(self):
        """Test building configuration for ChromaDB + sentence-transformers."""
        setup = InteractiveSetup(quiet=False)
        vector_config = {"path": "./vectordb"}
        embedding_config = {"model": "all-MiniLM-L6-v2", "device": "cpu"}

        config = setup._build_config(
            "chromadb",
            vector_config,
            "sentence-transformers",
            embedding_config
        )

        assert config["vectorStore"]["provider"] == "chromadb"
        assert config["vectorStore"]["chromadb"]["path"] == "./vectordb"
        assert config["embedding"]["provider"] == "sentence-transformers"
        assert config["embedding"]["sentenceTransformers"]["model"] == "all-MiniLM-L6-v2"

    def test_build_config_pinecone_openai(self):
        """Test building configuration for Pinecone + OpenAI."""
        setup = InteractiveSetup(quiet=False)
        vector_config = {
            "apiKey": "${PINECONE_API_KEY}",
            "environment": "us-east-1-aws",
            "indexName": "raggy-index"
        }
        embedding_config = {
            "apiKey": "${OPENAI_API_KEY}",
            "model": "text-embedding-3-small"
        }

        config = setup._build_config(
            "pinecone",
            vector_config,
            "openai",
            embedding_config
        )

        assert config["vectorStore"]["provider"] == "pinecone"
        assert config["vectorStore"]["pinecone"]["apiKey"] == "${PINECONE_API_KEY}"
        assert config["vectorStore"]["pinecone"]["dimension"] == 1536
        assert config["embedding"]["provider"] == "openai"
        assert config["embedding"]["openai"]["model"] == "text-embedding-3-small"

    def test_is_env_var_placeholder_valid(self):
        """Test environment variable placeholder detection with valid format."""
        setup = InteractiveSetup(quiet=False)
        assert setup._is_env_var_placeholder("${API_KEY}") is True
        assert setup._is_env_var_placeholder("${SUPABASE_URL}") is True
        assert setup._is_env_var_placeholder("${OPENAI_API_KEY}") is True

    def test_is_env_var_placeholder_invalid(self):
        """Test environment variable placeholder detection with invalid format."""
        setup = InteractiveSetup(quiet=False)
        assert setup._is_env_var_placeholder("api_key_123") is False
        assert setup._is_env_var_placeholder("$API_KEY") is False
        assert setup._is_env_var_placeholder("{API_KEY}") is False
        assert setup._is_env_var_placeholder("") is False

    def test_validate_url_valid(self):
        """Test URL validation with valid URLs."""
        setup = InteractiveSetup(quiet=False)
        assert setup._validate_url("https://example.com") is True
        assert setup._validate_url("http://localhost:8000") is True
        assert setup._validate_url("${SUPABASE_URL}") is True

    def test_validate_url_invalid(self):
        """Test URL validation with invalid URLs."""
        setup = InteractiveSetup(quiet=False)
        assert setup._validate_url("not-a-url") is False
        assert setup._validate_url("ftp://example.com") is False
        assert setup._validate_url("") is False

    def test_validate_api_key_valid(self):
        """Test API key validation with valid keys."""
        setup = InteractiveSetup(quiet=False)
        assert setup._validate_api_key("sk-1234567890") is True
        assert setup._validate_api_key("${OPENAI_API_KEY}") is True
        assert setup._validate_api_key("a" * 50) is True

    def test_validate_api_key_invalid(self):
        """Test API key validation with invalid keys."""
        setup = InteractiveSetup(quiet=False)
        assert setup._validate_api_key("short") is False
        assert setup._validate_api_key("") is False
        assert setup._validate_api_key("abc") is False

    def test_confirm_setup_quiet_returns_true(self):
        """Test confirm setup in quiet mode returns True."""
        setup = InteractiveSetup(quiet=True)
        setup.config = {"vectorStore": {"provider": "chromadb"}}
        assert setup._confirm_setup() is True

    @patch('builtins.input', return_value='Y')
    def test_confirm_setup_yes(self, mock_input):
        """Test confirm setup with Yes response."""
        setup = InteractiveSetup(quiet=False)
        setup.config = {
            "vectorStore": {
                "provider": "chromadb",
                "chromadb": {"path": "./vectordb"}
            },
            "embedding": {
                "provider": "sentence-transformers",
                "sentenceTransformers": {"model": "all-MiniLM-L6-v2"}
            }
        }
        assert setup._confirm_setup() is True

    @patch('builtins.input', return_value='n')
    def test_confirm_setup_no(self, mock_input):
        """Test confirm setup with No response."""
        setup = InteractiveSetup(quiet=False)
        setup.config = {
            "vectorStore": {
                "provider": "chromadb",
                "chromadb": {"path": "./vectordb"}
            },
            "embedding": {
                "provider": "sentence-transformers",
                "sentenceTransformers": {"model": "all-MiniLM-L6-v2"}
            }
        }
        assert setup._confirm_setup() is False

    @patch('builtins.input', return_value='')
    def test_confirm_setup_default_yes(self, mock_input):
        """Test confirm setup with default (Enter) is Yes."""
        setup = InteractiveSetup(quiet=False)
        setup.config = {
            "vectorStore": {
                "provider": "chromadb",
                "chromadb": {"path": "./vectordb"}
            },
            "embedding": {
                "provider": "sentence-transformers",
                "sentenceTransformers": {"model": "all-MiniLM-L6-v2"}
            }
        }
        assert setup._confirm_setup() is True

    def test_write_config_creates_file(self, tmp_path):
        """Test write_config creates .raggy.json file."""
        setup = InteractiveSetup(quiet=True)
        setup.config = {
            "_comment": "Test config",
            "vectorStore": {"provider": "chromadb"}
        }

        config_file = tmp_path / ".raggy.json"
        setup.write_config(config_file)

        assert config_file.exists()
        with open(config_file, 'r') as f:
            saved_config = json.load(f)
        assert saved_config["vectorStore"]["provider"] == "chromadb"

    @patch('builtins.input', side_effect=KeyboardInterrupt())
    def test_run_handles_keyboard_interrupt(self, mock_input):
        """Test run handles Ctrl+C gracefully."""
        setup = InteractiveSetup(quiet=False)
        with pytest.raises(SystemExit):
            setup.run()

    @patch('builtins.input', side_effect=EOFError())
    def test_run_handles_eof_error(self, mock_input):
        """Test run handles Ctrl+D gracefully."""
        setup = InteractiveSetup(quiet=False)
        with pytest.raises(SystemExit):
            setup.run()

    def test_run_quiet_mode_completes(self):
        """Test run in quiet mode completes without user input."""
        setup = InteractiveSetup(quiet=True)
        with patch.object(setup, '_confirm_setup', return_value=True):
            config = setup.run()

        assert config is not None
        assert "vectorStore" in config
        assert "embedding" in config

    @patch('builtins.input', side_effect=['', '', '', 'Y'])
    def test_run_interactive_with_defaults(self, mock_input):
        """Test run in interactive mode with all defaults."""
        setup = InteractiveSetup(quiet=False)
        config = setup.run()

        assert config["vectorStore"]["provider"] == "chromadb"
        assert config["embedding"]["provider"] == "sentence-transformers"


def test_run_interactive_setup_success(tmp_path):
    """Test run_interactive_setup function with success."""
    with patch('raggy.setup.interactive.InteractiveSetup') as MockSetup:
        mock_instance = MockSetup.return_value
        mock_instance.run.return_value = {"test": "config"}
        mock_instance.config = {"test": "config"}
        mock_instance.quiet = True
        mock_instance.write_config = MagicMock()

        result = run_interactive_setup(quiet=True)

        assert result is True
        mock_instance.run.assert_called_once()
        mock_instance.write_config.assert_called_once()


def test_run_interactive_setup_keyboard_interrupt():
    """Test run_interactive_setup handles KeyboardInterrupt."""
    with patch('raggy.setup.interactive.InteractiveSetup') as MockSetup:
        mock_instance = MockSetup.return_value
        mock_instance.run.side_effect = KeyboardInterrupt()

        result = run_interactive_setup(quiet=False)

        assert result is False
