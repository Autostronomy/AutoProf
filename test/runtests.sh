echo "autoprof test"
autoprof test_config.py &> output_autoprof.txt
echo "forced test"
autoprof test_forced_config.py Forced.log &> output_forced.txt
echo "batch test"
autoprof test_batch_config.py Batch.log &> output_batch.txt
echo "decision tree test"
autoprof test_tree_config.py Tree.log &> output_tree.txt
echo "checking for errors (will be written below if any):"
grep ERROR *.log
echo "all done!"
