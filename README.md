# alienworlds
Final Year Project: Technosignatures in Kepler and TESS data...

## The Story So Far:

#### Get Data: 
Data is obtained through the stsci archive. There are many other available options as well. To customize the download the following link can be explored.
[STSCI Archive](https://archive.stsci.edu/pub/kepler/)

Check out 'ex_download.py', a script to download the DV time series for the kepler data, through the KOI table

#### Get Catalogs: 
The catalogs for the required kepler IDs are obtained through the exoplanet archive.

#### Get Binned Files: 
Once the raw files are obtained, they can be binned as per requirement of the Neural Network using 'preparing_data.py'. The options explored here are:
1. Phase folded local and global view lightcurves
2. Raw lightcurves of different bin sizes
3. Stitched transits

#### Get Training Sample:
The binned files are stored in a directory. Since creation of TS is time-consuming, the 'NN_training_sample.py' creates a handy csv file with the acumulation
of labels and lightcurves that can simply be imported to the NN currently in function.

#### View Plots:
To view the lightcurves in various stages of processing. 'plot_binned.py' is a useful script

### NN Architectures explored:
Three architectures are explored that show promising results:
1. CNN with local and global view input on phase folded LC
2. VAE with raw input
3. A take on the Inception-Resnet Module with raw input
