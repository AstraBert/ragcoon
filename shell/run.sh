eval "$(conda shell.bash hook)"

conda activate ragcoon
cd /app/
uvicorn main:app --host 0.0.0.0 --port 8000
conda deactivate
