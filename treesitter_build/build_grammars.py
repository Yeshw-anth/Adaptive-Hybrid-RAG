import os
import platform
from tree_sitter import Language

# Define the directory to store grammars, relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
GRAMMAR_DIR = os.path.join(script_dir, "tree-sitter-grammars")

def get_library_path():
    """Determines the library path based on the operating system."""
    # Get the project root directory (assuming this script is in a subdirectory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(project_root, "treesitter_build")
    os.makedirs(build_dir, exist_ok=True)
    
    system = platform.system()
    if system == "Windows":
        return os.path.join(build_dir, "languages.dll")
    elif system == "Linux":
        return os.path.join(build_dir, "languages.so")
    elif system == "Darwin":
        return os.path.join(build_dir, "languages.dylib")
    else:
        raise RuntimeError(f"Unsupported operating system: {system}")

# Determine the correct library path
LIBRARY_PATH = get_library_path()

# List of languages to support with their grammar git repositories
LANGUAGES = {
    "python": "https://github.com/tree-sitter/tree-sitter-python",
    "javascript": "https://github.com/tree-sitter/tree-sitter-javascript",
}

# Specify compatible grammar versions (tags or commits)
GRAMMAR_VERSIONS = {
    "python": "v0.19.0",
    "javascript": "v0.25.0",
}

def build_tree_sitter_grammars():
    """
    Clones language grammar repositories and compiles them into a single
    shared library for tree-sitter to use. If the library already exists,
    this function does nothing.
    """
    # Per your instruction, check if the library already exists before doing any work.
    if os.path.exists(LIBRARY_PATH):
        return

    os.makedirs(GRAMMAR_DIR, exist_ok=True)
    
    grammar_paths = []
    for lang, url in LANGUAGES.items():
        repo_path = os.path.join(GRAMMAR_DIR, f"tree-sitter-{lang}")
        version = GRAMMAR_VERSIONS.get(lang)
        
        if not os.path.exists(repo_path):
            print(f"Cloning {lang} grammar from {url}...")
            os.system(f"git clone {url} {repo_path}")
            if version:
                print(f"Checking out version {version} for {lang} grammar...")
                os.system(f"cd {repo_path} && git checkout {version}")
        else:
            print(f"Grammar for {lang} already cloned.")
            # Optional: ensure it's on the correct version even if it exists
            if version:
                print(f"Verifying and checking out version {version} for {lang} grammar...")
                os.system(f"cd {repo_path} && git checkout {version}")
        
        grammar_paths.append(repo_path)

    print(f"\nCompiling grammars into shared library at '{LIBRARY_PATH}'...")
    
    Language.build_library(
        # The name of the output library
        LIBRARY_PATH,
        # The paths to the grammar source directories
        grammar_paths
    )
    
    print("\nGrammar library built successfully.")