"""Streamlit Cloud entrypoint (default main-module name).

Delegates to telco_churn_streamlit_app so either filename works in Cloud settings.
"""

from telco_churn_streamlit_app import main

if __name__ == "__main__":
    main()
else:
    # When Streamlit imports this module as the script, run main immediately.
    main()
