# get images and merge/filter
python get_images_Maxar.py --disaster typhoon-mangkhut --dest ~/datalake/maxar/typhoon-mangkhut/raw --maxpre 20 --maxpost 20
python filter_images.py --mosaic True --country Philippines --ntl True --data ~/datalake/maxar/typhoon-mangkhut/raw --dest ~/datalake/maxar/typhoon-mangkhut/processed
# predict buildings
neo cover --raster ~/datalake/maxar/typhoon-mangkhut/processed/pre-event/*-ntl.tif --zoom 17 --out ~/datalake/maxar/typhoon-mangkhut/neo/cover.csv
neo tile --raster ~/datalake/maxar/typhoon-mangkhut/processed/pre-event/*-ntl.tif --zoom 17 --cover ~/datalake/maxar/typhoon-mangkhut/neo/cover.csv --config ~/neateo/config.toml --out ~/datalake/maxar/typhoon-mangkhut/neo/images --format tif
mkdir ~/datalake/maxar/typhoon-mangkhut/neo/predictions
neo predict --config ~/neateo/config.toml --dataset ~/datalake/maxar/typhoon-mangkhut/neo --cover ~/datalake/maxar/typhoon-mangkhut/neo/cover.csv --checkpoint ~/datalake/neateo-models/neat-fullxview-epoch75.pth --out ~/datalake/maxar/typhoon-mangkhut/neo/predictions --metatiles --keep_borders
neo vectorize --masks ~/datalake/maxar/typhoon-mangkhut/neo/predictions --type Building --config ~/neateo/config.toml --out ~/datalake/maxar/typhoon-mangkhut/processed/buildings.geojson
python filter_buildings.py
# classify damage
python prepare_data_for_caladrius.py --data ~/datalake/maxar/typhoon-mangkhut/processed --dest ~/datalake/maxar/typhoon-mangkhut/caladrius
python ~/caladrius-master/caladrius/run.py --run-name caladrius_2020 --data-path ~/datalake/maxar/typhoon-mangkhut-2/caladrius --model-path ~/datalake/caladrius-models/all_wind_class_10_epochs-input_size_32-learning_rate_0.001-batch_size_16/best_model_wts.pkl --checkpoint-path ~/datalake/maxar/typhoon-mangkhut-2/caladrius/runs --input-type classification --inference
# create final geodataframe
python final_layer.py