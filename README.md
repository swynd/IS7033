# IS7033 Project - Drug Discovery Model

### Introduction
The [IDG-DREAM Challenge](https://www.synapse.org/#!Synapse:syn15667962) [1] was created to bring in competitors to design models to predict drug capabilities to provide biomedical scientists some computational insight for valuable targets for further research. The focus of this competition is on drugs that inhibit Kinase protein activity within the biological cell. Kinase proteins main functions are to interact with other molecules (usually proteins) to impact their enzyme activity, location within the cell, or how they interact with other proteins [2]. Protein function is driven almost exclusively by the molecular sequence and structure of the amino acids that make up that protein. Drug molecules are able to interact with the protein and prevent the protein from performing the functions for which the protein is usually responsible. 

Because the interaction between the drug molecule and the protein is able to impact the protein function, it is very important for sceintists to understand the mechanics of such an interaction. One method of quantifying this interaction is using the dissociation constant Kd, which is represented by the following formula:

Kd = [Concentration of protein] . [Concentration of drug] / [Concentration of protein-drug complex]

Using this metric, drug-protein interactions with a very high Kd do not bind very strongly, but drug-protein interaction with very low Kd have a higher affinity for eachother and therefore are more likely to bond strongly. This high affinity can often mean that the drug will have some ability to bind to and disrupt the protein function. The model discussed in this paper takes a dataset created by the IDG-DREAM Challenge team with Kd values for 47,000+ drug-protein interactions, including the amino acid sequence of the protein and a binary representation of the drug molecule. 

### Methodology
When the drug and protein interact, because drug molecules are not much larger than one or two amino acids, it is very likely that the drug will interact with two or three consecutive amino acids in the protein, and maybe four consecutive amino acids in some extreme cases. To create features to reflect these consecutive amino acids, we pulled pairs of amino acids that were consecutive in the sequence, or were separated by one or two amino acids. These pairs were then converted to one-hot encoding (224 columns for each of the three sets of pairs) to feed in to the model and represented the protein sequence amino acids. The drugs did no require any transformation, as the drugs were stored as binary numerals, 920 digits long, which were representative of the atoms, bonds, and strucutre of the drugs. The final training dataset had 1,592 columns with 47,235 drug-protein interactions. This data set was split into a train and test set, with 85% of the interactions being used to train, and 15% being held back to test the model.

It was unclear what architecture would work best for this data set, a strategy was devised to test as many parameters and as many model architectures as possible. To be able to implement these parameters and architectures, a configuration file structure was created that could house all of the parameters, count of hidden layers, and other relevant model characteristics. The configuration files have the following structure in a json format:
```python
params = {
    'batch_size': int,          # batch size used during each step              100, 200, 300, 400 
    'epochs': int,              # number of epochs to run the model             50, 100, 150
    'learning_rate': float,     # learning rate to feed into the optimizer      0.001, 0.0001
    'optimizer': string,        # which optimizer to use                        'adam', 'rmsprop'
    'hidden_layers': int,       # number of hidden layers to use                2, 3, 4, 5, 6, 7
    'dropout': float,           # ratio of nodes to drop after first  layer     0.2, 0.4, 0.6
    'hidden_act': string,       # activation function for hidden layers         'relu', 'tanh', 'sigmoid'
    'output_act': string,       # activation function for output layer          'relu', 'sigmoid'
    'scaling': float            # ratio to scale node count in each layer       0.5, 0.6
}
```
Creating the configuration files made it very simple to tweak one or a few parameters at a time to evaluate their impact on the training of the model. The range of parameters evaluated in each of the parameters is seen to the right of the description above. 

### Results
After evaluating many different permutations of these parameters, the model with the lowest mean squared error was identified, and had the following parameters:
```python
batch_size:     300
epochs:         50
learning_rate:  0.001
optimizer:      'adam'
hidden_layers:  2
dropout:        0.2
hidden_act:     'tanh'
output_act:     'relu'
scaling:        0.5
```
The mean squared error of this model was 0.0521. After trying many different values for the number of hidden layers, it was clear that fewer layers was more efficient for the training of this model, as the optimal number of hidden layers was two. Adding dropout after the first layer did increase the performance of the model, the ratio of 0.2 was sufficient to impact the model. 

### Conclusion
This model was able to achieve a considerable accuracy when predicting the binding affinity between various drugs and proteins. All things considered, the problem is a relatively simple one, and only required a simple model to achieve valuable results. However, to improve this model further, and to obtain more nuanced insight into how and why drug molecules interact with proteins, it would be valuable to expand the scope of this model to look at protein structure more completely.
One potential strategy to evaluate protein structure is to feed the sequences into a recurrent neural network, to allow for more interactions between amino acids that are further apart than two or three spaces apart. Furthermore, an RNN would allow for compounding of impacts when three or four amino acids are present in a sequence. Another potential strategy for evaluating the structure of a protein is to treat protein sequence like a sentence, and evaluate the amino acids or groups of amino acids as words in the sentence, and evaluate "word" embeddings based on the population of protein sequences available. The upside of these analyses goes way beyond understanding interactions with drug molecules, and is a very exciting field to investigate further, and will be the main focus of study for my independent study for the foreseeable future. 

### References

[1] [IDG-DREAM Challenge website - https://www.synapse.org/#!Synapse:syn15667962](https://www.synapse.org/#!Synapse:syn15667962)

[2] [Cichonska et al. Computational-experimental approach to drug-target interaction mapping: A case study on kinase inhibitors.](https://www.ncbi.nlm.nih.gov/pubmed/28787438) PLoS Comput Biology 2017;13:e1005678.

[3] [Dissociation constant](https://en.wikipedia.org/wiki/Dissociation_constant) Wikipedia
