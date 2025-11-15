"""Interactive configuration questionnaire for raggy init command."""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Union


class InteractiveSetup:
    """Interactive setup questionnaire for raggy configuration."""

    # Vector store configurations
    VECTOR_STORES = {
        "1": {
            "name": "ChromaDB",
            "description": "Local, free, no setup required",
            "provider": "chromadb",
            "default": True,
        },
        "2": {
            "name": "Pinecone",
            "description": "Cloud-managed, requires API key",
            "provider": "pinecone",
            "default": False,
        },
        "3": {
            "name": "Supabase",
            "description": "PostgreSQL + pgvector, requires project",
            "provider": "supabase",
            "default": False,
        },
    }

    # Embedding provider configurations
    EMBEDDING_PROVIDERS = {
        "1": {
            "name": "sentence-transformers",
            "description": "Local, free, runs on CPU/GPU",
            "provider": "sentence-transformers",
            "default": True,
        },
        "2": {
            "name": "OpenAI",
            "description": "Cloud API, requires API key, higher quality",
            "provider": "openai",
            "default": False,
        },
    }

    # Sentence transformers model presets
    ST_MODELS = {
        "1": {
            "name": "all-MiniLM-L6-v2",
            "description": "Fast, balanced, 384 dimensions",
            "model": "all-MiniLM-L6-v2",
            "dimension": 384,
            "default": True,
        },
        "2": {
            "name": "all-mpnet-base-v2",
            "description": "Accurate, slower, 768 dimensions",
            "model": "all-mpnet-base-v2",
            "dimension": 768,
            "default": False,
        },
        "3": {
            "name": "paraphrase-multilingual-MiniLM-L12-v2",
            "description": "Multilingual, 384 dimensions",
            "model": "paraphrase-multilingual-MiniLM-L12-v2",
            "dimension": 384,
            "default": False,
        },
    }

    # OpenAI model options
    OPENAI_MODELS = {
        "1": {
            "name": "text-embedding-3-small",
            "description": "1536-dim, fast",
            "model": "text-embedding-3-small",
            "dimension": 1536,
            "default": True,
        },
        "2": {
            "name": "text-embedding-3-large",
            "description": "3072-dim, accurate, expensive",
            "model": "text-embedding-3-large",
            "dimension": 3072,
            "default": False,
        },
    }

    def __init__(self, quiet: bool = False):
        """Initialize interactive setup.

        Args:
            quiet: If True, suppress output (non-interactive mode)

        """
        self.quiet = quiet
        self.config: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """Run interactive setup questionnaire.

        Returns:
            Dict containing configuration selections

        Raises:
            KeyboardInterrupt: If user cancels with Ctrl+C

        """
        try:
            self._print_welcome()

            # Step 1: Select vector store
            vector_store = self._select_vector_store()

            # Step 2: Select embedding provider
            embedding_provider = self._select_embedding_provider()

            # Step 3: Select model based on provider
            if embedding_provider == "sentence-transformers":
                model_info = self._select_st_model()
            else:  # openai
                model_info = self._select_openai_model()

            # Step 4: Collect provider-specific configuration
            vector_config = self._collect_vector_store_config(vector_store)
            embedding_config = self._collect_embedding_config(
                embedding_provider, model_info
            )

            # Build complete configuration
            self.config = self._build_config(
                vector_store, vector_config, embedding_provider, embedding_config
            )

            # Step 5: Show summary and confirm
            if not self._confirm_setup():
                print("\nSetup cancelled.")
                sys.exit(0)

            return self.config

        except (EOFError, KeyboardInterrupt):
            print("\n\nSetup cancelled by user.")
            sys.exit(0)

    def _print_welcome(self) -> None:
        """Print welcome message."""
        if self.quiet:
            return

        print("\n🚀 Raggy Interactive Setup")
        print("\nWelcome! Let's configure your RAG system.\n")

    def _select_vector_store(self) -> str:
        """Prompt user to select vector store provider.

        Returns:
            str: Selected provider name (chromadb, pinecone, supabase)

        """
        if self.quiet:
            return "chromadb"

        print("📦 Vector Store (where embeddings are stored):")
        for key, config in self.VECTOR_STORES.items():
            default_marker = " [default]" if config["default"] else ""
            print(f"  {key}. {config['name']} - {config['description']}{default_marker}")

        selection = self._get_input_with_default(
            "\nSelect vector store", default="1"
        )

        if selection not in self.VECTOR_STORES:
            print(f"Invalid selection '{selection}', using default (ChromaDB)")
            selection = "1"

        return self.VECTOR_STORES[selection]["provider"]

    def _select_embedding_provider(self) -> str:
        """Prompt user to select embedding provider.

        Returns:
            str: Selected provider name (sentence-transformers, openai)

        """
        if self.quiet:
            return "sentence-transformers"

        print("\n🧠 Embedding Model (how text is converted to vectors):")
        for key, config in self.EMBEDDING_PROVIDERS.items():
            default_marker = " [default]" if config["default"] else ""
            print(f"  {key}. {config['name']} - {config['description']}{default_marker}")

        selection = self._get_input_with_default(
            "\nSelect embedding provider", default="1"
        )

        if selection not in self.EMBEDDING_PROVIDERS:
            print(f"Invalid selection '{selection}', using default (sentence-transformers)")
            selection = "1"

        return self.EMBEDDING_PROVIDERS[selection]["provider"]

    def _select_st_model(self) -> Dict[str, Any]:
        """Prompt user to select sentence-transformers model.

        Returns:
            Dict containing model name, description, and dimension

        """
        if self.quiet:
            return self.ST_MODELS["1"]

        print("\n📊 Sentence Transformers Model:")
        for key, config in self.ST_MODELS.items():
            default_marker = " [default]" if config["default"] else ""
            print(f"  {key}. {config['name']} - {config['description']}{default_marker}")

        selection = self._get_input_with_default("\nSelect model", default="1")

        if selection not in self.ST_MODELS:
            print(f"Invalid selection '{selection}', using default (all-MiniLM-L6-v2)")
            selection = "1"

        return self.ST_MODELS[selection]

    def _select_openai_model(self) -> Dict[str, Any]:
        """Prompt user to select OpenAI model.

        Returns:
            Dict containing model name, description, and dimension

        """
        if self.quiet:
            return self.OPENAI_MODELS["1"]

        print("\n📊 OpenAI Embedding Model:")
        for key, config in self.OPENAI_MODELS.items():
            default_marker = " [default]" if config["default"] else ""
            print(f"  {key}. {config['name']} - {config['description']}{default_marker}")

        selection = self._get_input_with_default("\nSelect model", default="1")

        if selection not in self.OPENAI_MODELS:
            print(f"Invalid selection '{selection}', using default (text-embedding-3-small)")
            selection = "1"

        return self.OPENAI_MODELS[selection]

    def _collect_vector_store_config(self, provider: str) -> Dict[str, Any]:
        """Collect provider-specific vector store configuration.

        Args:
            provider: Vector store provider name

        Returns:
            Dict containing provider-specific configuration

        """
        if provider == "chromadb":
            return {"path": "./vectordb"}

        elif provider == "pinecone":
            return self._collect_pinecone_config()

        elif provider == "supabase":
            return self._collect_supabase_config()

        return {}

    def _collect_pinecone_config(self) -> Dict[str, Any]:
        """Collect Pinecone configuration.

        Returns:
            Dict containing Pinecone API key, environment, and index name

        """
        if self.quiet:
            return {
                "apiKey": "${PINECONE_API_KEY}",
                "environment": "us-east-1-aws",
                "indexName": "raggy-index",
            }

        print("\n⚙️  Pinecone Configuration:")
        print("Tip: You can use environment variable placeholders like ${PINECONE_API_KEY}")

        # Get and validate API key
        while True:
            api_key = self._get_input_with_default(
                "Pinecone API key", default="${PINECONE_API_KEY}"
            )
            if self._validate_api_key(api_key):
                break
            print("  Warning: API key seems invalid (should be 10+ characters or ${VAR})")
            retry = input("  Continue anyway? [y/N]: ").strip().lower()
            if retry in ('y', 'yes'):
                break

        environment = self._get_input_with_default(
            "Pinecone environment (e.g., us-east-1-aws)", default="us-east-1-aws"
        )

        index_name = self._get_input_with_default(
            "Pinecone index name", default="raggy-index"
        )

        return {
            "apiKey": api_key,
            "environment": environment,
            "indexName": index_name,
        }

    def _collect_supabase_config(self) -> Dict[str, Any]:
        """Collect Supabase configuration.

        Returns:
            Dict containing Supabase URL and API key

        """
        if self.quiet:
            return {
                "url": "${SUPABASE_URL}",
                "apiKey": "${SUPABASE_ANON_KEY}",
            }

        print("\n⚙️  Supabase Configuration:")
        print("Tip: You can use environment variable placeholders like ${SUPABASE_URL}")

        # Get and validate URL
        while True:
            url = self._get_input_with_default(
                "Supabase project URL", default="${SUPABASE_URL}"
            )
            if self._validate_url(url):
                break
            print("  Warning: URL seems invalid (should start with https:// or be ${VAR})")
            retry = input("  Continue anyway? [y/N]: ").strip().lower()
            if retry in ('y', 'yes'):
                break

        # Get and validate API key
        while True:
            api_key = self._get_input_with_default(
                "Supabase API key (anon)", default="${SUPABASE_ANON_KEY}"
            )
            if self._validate_api_key(api_key):
                break
            print("  Warning: API key seems invalid (should be 10+ characters or ${VAR})")
            retry = input("  Continue anyway? [y/N]: ").strip().lower()
            if retry in ('y', 'yes'):
                break

        return {
            "url": url,
            "apiKey": api_key,
        }

    def _collect_embedding_config(
        self, provider: str, model_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect provider-specific embedding configuration.

        Args:
            provider: Embedding provider name
            model_info: Selected model information

        Returns:
            Dict containing provider-specific configuration

        """
        if provider == "sentence-transformers":
            return {
                "model": model_info["model"],
                "device": "cpu",
            }

        elif provider == "openai":
            return self._collect_openai_config(model_info)

        return {}

    def _collect_openai_config(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Collect OpenAI configuration.

        Args:
            model_info: Selected OpenAI model information

        Returns:
            Dict containing OpenAI API key and model name

        """
        if self.quiet:
            return {
                "apiKey": "${OPENAI_API_KEY}",
                "model": model_info["model"],
            }

        print("\n⚙️  OpenAI Configuration:")
        print("Tip: You can use environment variable placeholders like ${OPENAI_API_KEY}")

        # Get and validate API key
        while True:
            api_key = self._get_input_with_default(
                "OpenAI API key", default="${OPENAI_API_KEY}"
            )
            if self._validate_api_key(api_key):
                break
            print("  Warning: API key seems invalid (should be 10+ characters or ${VAR})")
            retry = input("  Continue anyway? [y/N]: ").strip().lower()
            if retry in ('y', 'yes'):
                break

        return {
            "apiKey": api_key,
            "model": model_info["model"],
        }

    def _get_embedding_dimension(
        self, provider: str, embedding_config: Dict[str, Any]
    ) -> int:
        """Get embedding dimension from provider and model configuration.

        Args:
            provider: Embedding provider name
            embedding_config: Embedding configuration dict

        Returns:
            int: Embedding dimension

        """
        if provider == "sentence-transformers":
            model_name = embedding_config.get("model", "")
            for model_data in self.ST_MODELS.values():
                if model_data["model"] == model_name:
                    return model_data["dimension"]
            return 384  # Default for sentence-transformers

        elif provider == "openai":
            model_name = embedding_config.get("model", "")
            for model_data in self.OPENAI_MODELS.values():
                if model_data["model"] == model_name:
                    return model_data["dimension"]
            return 1536  # Default for OpenAI

        return 384  # Fallback default

    def _build_config(
        self,
        vector_store: str,
        vector_config: Dict[str, Any],
        embedding_provider: str,
        embedding_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build complete configuration from selections.

        Args:
            vector_store: Selected vector store provider
            vector_config: Vector store configuration
            embedding_provider: Selected embedding provider
            embedding_config: Embedding configuration

        Returns:
            Dict containing complete configuration

        """
        config: Dict[str, Any] = {
            "_comment": "Raggy Configuration - Generated by interactive setup",
            "vectorStore": {
                "provider": vector_store,
            },
            "embedding": {
                "provider": embedding_provider,
            },
        }

        # Determine dimension from embedding model
        dimension = self._get_embedding_dimension(embedding_provider, embedding_config)

        # Add provider-specific vector store config
        if vector_store == "chromadb":
            config["vectorStore"]["chromadb"] = vector_config
        elif vector_store == "pinecone":
            config["vectorStore"]["pinecone"] = vector_config
            config["vectorStore"]["pinecone"]["dimension"] = dimension
        elif vector_store == "supabase":
            config["vectorStore"]["supabase"] = vector_config
            config["vectorStore"]["supabase"]["dimension"] = dimension

        # Add provider-specific embedding config
        if embedding_provider == "sentence-transformers":
            config["embedding"]["sentenceTransformers"] = embedding_config
        elif embedding_provider == "openai":
            config["embedding"]["openai"] = embedding_config

        return config

    def _confirm_setup(self) -> bool:
        """Show configuration summary and confirm with user.

        Returns:
            bool: True if user confirms, False otherwise

        """
        if self.quiet:
            return True

        print("\n✅ Configuration Summary:")

        # Show vector store
        vector_provider = self.config["vectorStore"]["provider"]
        if vector_provider == "chromadb":
            path = self.config["vectorStore"]["chromadb"]["path"]
            print(f"  Vector Store: ChromaDB (local at {path})")
        elif vector_provider == "pinecone":
            env = self.config["vectorStore"]["pinecone"]["environment"]
            index = self.config["vectorStore"]["pinecone"]["indexName"]
            print(f"  Vector Store: Pinecone ({env}, index: {index})")
        elif vector_provider == "supabase":
            url = self.config["vectorStore"]["supabase"]["url"]
            print(f"  Vector Store: Supabase ({url})")

        # Show embedding
        embedding_provider = self.config["embedding"]["provider"]
        if embedding_provider == "sentence-transformers":
            model = self.config["embedding"]["sentenceTransformers"]["model"]
            print(f"  Embedding: sentence-transformers/{model}")
        elif embedding_provider == "openai":
            model = self.config["embedding"]["openai"]["model"]
            print(f"  Embedding: OpenAI/{model}")

        confirm = self._get_input_with_default(
            "\nProceed with setup? [Y/n]", default="Y"
        )

        return confirm.lower() in ("y", "yes", "")

    def _get_input_with_default(self, prompt: str, default: str) -> str:
        """Get user input with a default value.

        Args:
            prompt: Prompt to display to user
            default: Default value if user presses Enter

        Returns:
            str: User input or default value

        """
        full_prompt = f"{prompt} [{default}]: "
        user_input = input(full_prompt).strip()
        return user_input if user_input else default

    def _is_env_var_placeholder(self, value: str) -> bool:
        """Check if value is an environment variable placeholder.

        Args:
            value: Value to check

        Returns:
            bool: True if value is like ${VAR_NAME}

        """
        return bool(re.match(r'^\$\{[A-Z_][A-Z0-9_]*\}$', value))

    def _validate_url(self, url: str) -> bool:
        """Validate URL format.

        Args:
            url: URL to validate

        Returns:
            bool: True if URL is valid or is an env var placeholder

        """
        # Allow environment variable placeholders
        if self._is_env_var_placeholder(url):
            return True

        # Basic URL validation (http:// or https://)
        url_pattern = r'^https?://.+'
        return bool(re.match(url_pattern, url))

    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key format (basic check).

        Args:
            api_key: API key to validate

        Returns:
            bool: True if API key looks valid or is an env var placeholder

        """
        # Allow environment variable placeholders
        if self._is_env_var_placeholder(api_key):
            return True

        # Basic check: non-empty and reasonable length
        return len(api_key) >= 10

    def write_config(self, config_path: Union[str, Path] = ".raggy.json") -> None:
        """Write configuration to file.

        Args:
            config_path: Path to configuration file

        """
        config_file = Path(config_path)

        try:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)

            if not self.quiet:
                print(f"\n✅ Created {config_path}")

        except (OSError, PermissionError) as e:
            print(f"ERROR: Failed to write configuration file: {e}")
            sys.exit(1)


def run_interactive_setup(quiet: bool = False) -> bool:
    """Run interactive setup and create .raggy.json configuration.

    Args:
        quiet: If True, use defaults without prompting

    Returns:
        bool: True if setup completed successfully

    """
    setup = InteractiveSetup(quiet=quiet)

    try:
        setup.run()
        setup.write_config()
        return True

    except (KeyboardInterrupt, EOFError):
        return False
