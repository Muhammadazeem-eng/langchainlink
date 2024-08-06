from dotenv import main
import os

updateEnv = main.dotenv_values()
os.environ.update(updateEnv)
main.load_dotenv()
