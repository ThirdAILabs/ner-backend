import sys
import logging

# redirect stdout to the stderr stream
sys.stdout = sys.stderr

# set the logging level to ERROR
logging.disable(logging.ERROR)
