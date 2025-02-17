cd agent && tar -czf submission.tar.gz *
mv submission.tar.gz ../submission.tar.gz
cd .. && kaggle competitions submit -c lux-ai-season-3 -f submission.tar.gz -m "submit"