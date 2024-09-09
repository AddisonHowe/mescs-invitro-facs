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
	# Version 1: No testing data; wrong timeshift 2.0 for dec 2; signal rate 10.
	# python script4a_dimred_pca.py -k facs_v1 -d 1 -p 0
	# python script4a_dimred_pca.py -k facs_v1 -d 1 --fit_on_subset -p 0
	# python script4a_dimred_pca.py -k facs_v1 -d 1 -nv -p 0
	# python script4a_dimred_pca.py -k facs_v1 -d 1 --fit_on_subset -nv -p 0
	# python script4a_dimred_pca.py -k facs_v1 -d 2 -t 2.0 -p 0
	# python script4a_dimred_pca.py -k facs_v1 -d 2 -t 2.0 --fit_on_subset -p 0
	# python script4a_dimred_pca.py -k facs_v1 -d 2 -t 2.0 -nv -p 0
	# python script4a_dimred_pca.py -k facs_v1 -d 2 -t 2.0 --fit_on_subset -nv -p 0

	# Version 2: No testing data; wrong timeshift 2.0 for dec 2; signal rate 1000.
	# python script4a_dimred_pca.py -k facs_v2 -d 1 -p 0
	# python script4a_dimred_pca.py -k facs_v2 -d 1 --fit_on_subset -p 0
	# python script4a_dimred_pca.py -k facs_v2 -d 1 -nv -p 0
	# python script4a_dimred_pca.py -k facs_v2 -d 1 --fit_on_subset -nv -p 0
	# python script4a_dimred_pca.py -k facs_v2 -d 2 -t 2.0 -p 0
	# python script4a_dimred_pca.py -k facs_v2 -d 2 -t 2.0 --fit_on_subset -p 0
	# python script4a_dimred_pca.py -k facs_v2 -d 2 -t 2.0 -nv -p 0
	# python script4a_dimred_pca.py -k facs_v2 -d 2 -t 2.0 --fit_on_subset -nv -p 0

	# Version 3: Testing data; wrong timeshift 2.0 for dec 2; signal rate 1000.
	python script4a_dimred_pca.py -k facs_v3 -d 1 --fit_on_subset
	python script4a_dimred_pca.py -k facs_v3 -d 2 -t 2.0 --fit_on_subset
	# python script4a_dimred_pca.py -k facs_v3 -d 1
	# python script4a_dimred_pca.py -k facs_v3 -d 2 -t 2.0

	# Version 4: Testing data; timeshift for dec 2; signal rate 1000.
	python script4a_dimred_pca.py -k facs_v4 -d 2 --fit_on_subset
	# python script4a_dimred_pca.py -k facs_v4 -d 2
	# python script4a_dimred_pca.py -k facs_v4 -d 2 -nv
	# python script4a_dimred_pca.py -k facs_v4 -d 2 --fit_on_subset -nv

run_script4b:
	python script4b_dimred_nmf.py -d 1 --fit_on_subset
	python script4b_dimred_nmf.py -d 2 --fit_on_subset
	# python script4b_dimred_nmf.py -d 1
	# python script4b_dimred_nmf.py -d 2

run_all:
	make run_script1
	make run_script1b
	make run_script2
	make run_script2a
	make run_script3a
	make run_script3b
	make run_script4a
	# make run_script4b
	
generate_all_ai_files:
	sh scripting/generate_fig2a_1.sh
	sh scripting/generate_fig2a_2.sh
	sh scripting/generate_fig3a_1.sh
	sh scripting/generate_fig3a_2.sh
	sh scripting/generate_fig3b_1.sh
	sh scripting/generate_fig3b_2.sh