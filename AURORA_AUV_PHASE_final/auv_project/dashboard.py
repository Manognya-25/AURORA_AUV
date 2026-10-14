"""Dashboard entry point. Run with: python dashboard.py

Options: --host / --port (environment: DASHBOARD_HOST / DASHBOARD_PORT). The server refuses to start on a port
another dashboard is already serving — see dashboard/serve.py for why that guard exists.
"""
import sys

from dashboard.serve import main

if __name__ == "__main__":
    sys.exit(main())
