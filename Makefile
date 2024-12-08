run_script1:
	python script1_preprocessing.py

run_script1b:
	python script1b_signal_plots.py

run_script2:
	python script2_clustering.py

run_script2a:
	python script2a_postcluster_plots.py

run_script3a:
	python script3a_isolate1.py

run_script3b:
	python script3b_isolate2.py

run_script4a:
	# facs_dec1_v1 --- Version 3: wrong timeshift 2.0 for dec 2; signal rate 1000.
	python script4a_dimred_pca.py -k facs_v3 -d 1 --fit_on_subset -p 0
	
	# facs_dec2_v1 --- Version 4: timeshift for dec 2; signal rate 1000.
	python script4a_dimred_pca.py -k facs_v4 -d 2 --fit_on_subset -p 0  

	# facs_dec1_v2 --- Version 5: timeshift for dec 2; signal rate 1000, log-normalized
	python script4a_dimred_pca.py -k facs_v5 -d 1 --fit_on_subset --log_normalize -p 0
	
	# facs_dec2_v2 --- Version 5: timeshift for dec 2; signal rate 1000, log-normalized
	python script4a_dimred_pca.py -k facs_v5 -d 2 --fit_on_subset --log_normalize -p 0

	# facs_dec1_v3 --- Version 5: timeshift for dec 2; signal rate 1000, logicle
	python script4a_dimred_pca.py -k facs_v5 -d 1 --fit_on_subset --logicle -p 0
	
	# facs_dec2_v3 --- Version 5: timeshift for dec 2; signal rate 1000, logicle
	python script4a_dimred_pca.py -k facs_v5 -d 2 --fit_on_subset --logicle -p 0

	# facs_dec1_v4 --- Version 5: timeshift for dec 2; signal rate 1000
	python script4a_dimred_pca.py -k facs_v5 -d 1 --fit_on_subset -p 0

run_script4b:
	# facs_dec1_v4 --- Version 5: timeshift for dec 2; signal rate 1000, logicle
	python script4b_dimred_nmf.py -k facs_v5 -d 1 --fit_on_subset --logicle -p 0
	
	# facs_dec2_v4 --- Version 5: timeshift for dec 2; signal rate 1000, logicle
	python script4b_dimred_nmf.py -k facs_v5 -d 2 --fit_on_subset --logicle -p 0

run_all:
	make run_script1
	make run_script1b
	make run_script2
	make run_script2a
	make run_script3a
	make run_script3b
	make run_script4a
	make run_script4b
	
generate_all_ai_files:
	sh scripting/generate_fig2a_1.sh
	sh scripting/generate_fig2a_2.sh
	sh scripting/generate_fig3a_1.sh
	sh scripting/generate_fig3a_2.sh
	sh scripting/generate_fig3b_1.sh
	sh scripting/generate_fig3b_2.sh
