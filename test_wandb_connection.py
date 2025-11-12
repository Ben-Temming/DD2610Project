import os
import wandb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("WANDB_API_KEY")
if not api_key:
    print("ERROR: WandB API key NOT found in environment!")
else:
    print("WandB API key found. Trying to login...")
    try:
        wandb.login(key=api_key, relogin=True)  # Force login with the key
        print("Successfully logged into WandB!")
    except Exception as e:
        print(f"Failed to login to WandB: {e}")
