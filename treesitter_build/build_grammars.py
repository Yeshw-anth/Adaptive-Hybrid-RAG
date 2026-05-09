import os
import platform
import subprocess
import sys
from tree_sitter import Language

# Define the directory to store grammars, relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
GRAMMAR_DIR = os.path.join(script_dir, "tree-sitter-grammars")

def get_library_path():
    """Determines the library path based on the operating system."""
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

LIBRARY_PATH = get_library_path()

LANGUAGES = {
    "python": "https://github.com/tree-sitter/tree-sitter-python",
    "javascript": "https://github.com/tree-sitter/tree-sitter-javascript",
}

GRAMMAR_VERSIONS = {
    "python": "v0.19.0",
    "javascript": "v0.25.0",
}

def run_command(command, cwd=None):
    """Runs a shell command and handles errors."""
    print(f"Running command: '{command}' in '{cwd or os.getcwd()}'")
    try:
        result = subprocess.run(
            command,
            check=True,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}\nError: {e.stderr}", file=sys.stderr)
        raise

def build_tree_sitter_grammars():
    """
    Clones language grammar repositories and compiles them into a single
    shared library for tree-sitter to use, only if the library does not already exist.
    """
    if os.path.exists(LIBRARY_PATH):
        print(f"Grammar library '{LIBRARY_PATH}' already exists. Skipping build.")
        return

    print("Grammar library not found. Starting build process...")
    os.makedirs(GRAMMAR_DIR, exist_ok=True)
    
    grammar_paths = []
    for lang, url in LANGUAGES.items():
        repo_path = os.path.join(GRAMMAR_DIR, f"tree-sitter-{lang}")
        version = GRAMMAR_VERSIONS.get(lang)
        
        if not os.path.exists(repo_path):
            print(f"Cloning {lang} grammar from {url}...")
            run_command(f"git clone {url} {repo_path}")
        else:
            print(f"Grammar for {lang} already exists. Using existing clone.")

        if version:
            print(f"Fetching latest tags for {lang} grammar...")
            run_command("git fetch --tags --force", cwd=repo_path)
            print(f"Checking out version {version} for {lang} grammar...")
            run_command(f"git checkout {version}", cwd=repo_path)
        
        grammar_paths.append(repo_path)

    print(f"\nCompiling grammars into shared library at '{LIBRARY_PATH}'...")
    
    Language.build_library(
        LIBRARY_PATH,
        grammar_paths
    )
    
    print("\nGrammar library built successfully.")

if __name__ == "__main__":
    try:
        build_tree_sitter_grammars()
    except Exception as e:
        print(f"\n--- FATAL: An error occurred during grammar building: {e} ---", file=sys.stderr)
        sys.exit(1)