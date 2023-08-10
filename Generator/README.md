# Generating Dataset

### Make the enviroment
Required libraries:
```bash
python==3.7.15
filelock==3.8.0
h5py==3.7.0
hdf5==1.10.6
numpy==1.21.5
opencv-python==4.6.0
```
I used `python==3.7` and carla server version 0.9.13 for this project.

### Customize data generation process configuration
1. Changing number of frames to be recorded:
   - In `main.py` change the parameter `egos_to_run` to a desired number. Parameter `frames_to_record_one_ego` will change the ego car after defined frames. There are 5 different weather condition in each data take process; Therefore, total number of frames that will be recorded is equal to: 
   ```bash
   total number of frames:= egos_to_run * frames_to_record_one_ego * 5
   ```
2. Changing egg path of carla
  - In order to import carla in file `setting.py` change the egg path to:
  ```bash
  CARLA_EGG_PATH = "path/to/carla/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg"
  ```
   Also change this value in file `create_content_on_hdf5` in directory `utils/create_video_on_hdf5`, in order to convert the outputs to images.
3. Make the following directories
   ```bash
   data/
   utils/create_video_on_hdf5/generated_data/
   birdview_v3_cache/
   birdview_v3_cache/Carla
   birdview_v3_cache/Carla/Maps
   generated_data/
   generated_data/depth/
   generated_data/mask/
   generated_data/raw_pv/
   generated_data/filtered_pv/
   generated_data/birdview/
   generated_data/info/
   ```
4. Specify the town
   - Before running the generation process, always set the town in `CarlaWorld.py` initializer. Following list contains the towns that are available in carla 0.9.13 default:
   ```bash
   Town01_Opt
   Town02_Opt 
   Town03_Opt 
   Town04_Opt 
   Town05_Opt 
   Town10HD_Opt
   ```

### Runing the process
1. Run the carla server using this command in its specified directory.
   ```bash
   ./CarlaUE4.sh -prefernvidia -RenderOffScreen
   ```
2. Run the `main.py` file using following command
   ```bash
   python main.py <name_of_output_file> -ve 100
   ```
   which -ve flag specified number of vehicles to be spawned.
   The resulted hdf5 file will be saved at `data/` directory.
   if you want to store raw files instead of hdf5 format, you can run the following command:
   ```bash
      python main.py sample -ve 100 -save_format no_hdf5
   ```
   In this way, data will be saved in `generated_data/` in different directories.

### Translating the output file to information
1. Run the following command at `utils/` directory.
   ```bash
   python create_content_on_hdf5.py
   ```
   it will be saved desired images and information as json files in `generated_data` directory.

# Acknowledgement
This project is inspired by [carla-dataset-runner](https://github.com/AlanNaoto/carla-dataset-runner) and [carla-birdeye-view](https://github.com/deepsense-ai/carla-birdeye-view) repositories.