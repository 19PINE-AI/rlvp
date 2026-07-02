# Source before running validate_env.py / webarena_train.py.
# CONFIRMED-WORKING recipe (live-validated 2026-07-02 on shopping_admin task 77).
export PYTHONPATH=/home/ubuntu/benchmarks/webarena/ST-WebAgentBench:/home/ubuntu/rlvp
# shopping_admin URL MUST include /admin (goto() hits the admin login form).
export SHOPPING_ADMIN="http://localhost:7780/admin"  WA_SHOPPING_ADMIN="http://localhost:7780/admin"
export GITLAB="http://localhost:8023"                WA_GITLAB="http://localhost:8023"
export SUITECRM="http://localhost:8080"              WA_SUITECRM="http://localhost:8080"
export SHOPPING="" REDDIT="" WIKIPEDIA="" MAP="" HOMEPAGE="" IPA_HOME=""
# venv with browsergym.stwebagentbench + playwright chromium:
#   source /home/ubuntu/benchmarks/webarena/wa_venv/bin/activate
