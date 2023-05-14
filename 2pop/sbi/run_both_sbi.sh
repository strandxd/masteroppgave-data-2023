run_scripts()
{
  echo "Starting 2pop simple:"
  python 2pop_sbi_simple.py

  printf '\n'
  sleep 5

  echo "Starting 2pop embedding"
  python 2pop_sbi_embedding.py

  # Keep terminal open
  $SHELL
}

run_scripts