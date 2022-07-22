# graphcuts
Concretization and implementation of the graph cuts image segmentation algorithm proposed by [Yuri Boykov and Gareth Funka-Lea in 2006](https://link.springer.com/article/10.1007/s11263-006-7934-5)

## regional term
- Seeds given by user input are segmented using recursive thresholding, too avoid swallowing small segements in a coarse mean
- each segment is regarded as Gaussian distribution and a mean and standard deviation are calculated
- To obtain the correspondence of each pixel to the object or background, the similarity to each seed segment is determining using the segment mean and standard deviation. This result is weighted by the relative proportion of the segment size to the total amount of object/background seeds

## Examples
In the resources folder, you can find some examples from the [Berkeley Image Segmentation Dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/dataset/images.html) on which I tested the algorithm performance. In the koala subfolder, you can find example outputs for a koala picture. The output includes:
- The image colored in a scale from white to red. Red values indicate high feature differences between neighbouring pixels, white values the opposite.  
![regional term example](https://github.com/lukas-ruettgers/graphcuts/blob/main/resources/koala/boundary_2500_lambda10_1657533783.4815354.png?raw=true)
- The seeds colored by the mean feature value of the segment they have been assigned to by the thresholding algorithm  
![regional term example](https://github.com/lukas-ruettgers/graphcuts/blob/main/resources/koala/seeds_1657462747.5013611.png?raw=true)
- The image colored in a color scale between blue and red. The more red a pixel is, the higher is the likelyhood it belongs to the object. The more blue a pixel is, the higher is the likelyhood it belongs to the background.  
![regional term example](https://github.com/lukas-ruettgers/graphcuts/blob/main/resources/koala/regional_2sqrt1657524481.615033.png?raw=true)
- The final segmentation, where the background pixels are removed.  
![final segmentation example](https://github.com/lukas-ruettgers/graphcuts/blob/main/resources/koala/result_noise2500_lambda10_1658498193.243594.png?raw=true)
