# GAN-Based-Anomaly-Detection---CICIDS_2017-Dataset
GAN framework is trained on distribution of benign samples from CICIDS_2017 dataset and tested on both anomalous and benign samples based on recontruction loss and discriminator loss.

To run the code</br>
python3 main.py bigan cicids_2017 run --nb_epochs=<number_epochs> --w=<float between 0 and 1> --m=<'cross-e','fm'> --d=<int> --rd=<int>

