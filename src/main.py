import sys
from orchestration.pipeline import run_pipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/main.py <path_to_folder>")
        sys.exit(1)

    folder_path = sys.argv[1]

    print(f"Running HERMES Agentic on folder: {folder_path}")
    result = run_pipeline(folder_path)

    print("Pipeline completed.")
    print(result)


if __name__ == "__main__":
    main()
