SUBMISSION_FILES= model/func_impl.py \
		  data/data_parallel_preprocess.py \
		  mpi_wrapper/comm.py \
		  student_kernel.py \
		  discussion2-1.txt

REPORT_FILE= pa2_report.json


handin.tar: $(REPORT_FILE) $(SUBMISSION_FILES)
	tar cvf handin.tar $(SUBMISSION_FILES) $(REPORT_FILE)

pa2_report.json: generate_pa2_report.py student_submit.py student_kernel.py model/func_impl.py data/data_parallel_preprocess.py mpi_wrapper/comm.py discussion2-1.txt
	python3 generate_pa2_report.py

clean:
	rm -f *~ handin.tar pa2_report.json pa2_report.txt pa2_report.stamp
