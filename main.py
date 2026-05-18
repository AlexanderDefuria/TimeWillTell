
import os
from src.main import main

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

if __name__ == "__main__":
    main()

