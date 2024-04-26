import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from datapreprocess import generate_kmer_vector

# Assuming ComplexNN class definition is here (copy from the training script)
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)  # First layer with 1024 units
        self.fc2 = nn.Linear(1024, 2048)  # Second layer with 2048 units
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # ensure 'map_location' is set if you are switching devices
    label_encoder = checkpoint['label_encoder']
    num_classes = len(label_encoder.classes_)
    model = NN(input_size=4096, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, label_encoder

def predict_and_plot(model, vector_list, label_encoder):
    # Convert the Python list of integers to a NumPy array of type float32
    vector_np = np.array(vector_list, dtype=np.float32)

    # Convert the NumPy array to a PyTorch tensor and add a batch dimension
    vector_tensor = torch.tensor(vector_np).unsqueeze(0)  # Unsqueeze adds the batch dimension

    # Forward pass
    with torch.no_grad():
        outputs = model(vector_tensor)
        probabilities = F.softmax(outputs, dim=1).numpy().flatten()

    # Use Seaborn for plotting
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(x=[str(cls) for cls in label_encoder.classes_], y=probabilities, palette="Blues_d")
    plt.xticks(rotation=45, ha="right")  # Rotate labels and align right for better visibility
    plt.ylabel('Probability')
    plt.xlabel('Classes')
    plt.title('Probability Distribution of Classes')
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.show()

def predict_and_get_lists(model, vector_list, label_encoder):
    # Convert the Python list of integers to a NumPy array of type float32
    vector_np = np.array(vector_list, dtype=np.float32)
    
    # Convert the NumPy array to a PyTorch tensor and add a batch dimension
    vector_tensor = torch.tensor(vector_np).unsqueeze(0)  # Unsqueeze adds the batch dimension
    
    # Forward pass
    with torch.no_grad():
        outputs = model(vector_tensor)
        probabilities = F.softmax(outputs, dim=1).numpy().flatten()
    
    # Get the class labels
    class_labels = [str(cls) for cls in label_encoder.classes_]
    
    return probabilities.tolist(), class_labels

seq = "GAAACAAGATTTCAATTCTCAAATGAAGCTGGTTACACACAGAACATACACAATAACACAAATAGTTGTAGCGTAACCCCACTCAAAATTCTAAACCCTAGCATGCGCACCTATTATTTCGTAATTAAAAACAGCAGTCACAAAAGGTGAGTAAGATCGATATAGAGAGGGTACCTGATCTAGAGAGGGTACCTGATTGTGAAGGAGGAGAGTGAGTAAATGTTTGTGCTACTTTAGATAAACTGCATACCATGTTTTTAAAATTGTGTTCTTTGATTGACTGGAGAATCACGGAGTTGACTGCCGTTGAAACATTCGCCGTTGACTTTGTCGACGCCGGGCGGGATGCTAAGATGCGCCGTTAGTGTTGGCATATGATCTACATTTTTATTTAGATGATAAGTGCTGCATTTTTCATAGAGTTGATGAAAGGTCTCTTTTTGGGGATCTGCTTTTTCTTGAAGAGATGTTTTTTGAAAGCATAAGTTTAATACCAGATTATTTATATTATTATAAATAATTTGTGTAAACGCATCTTAGGTGGATCTTTAGACTGTAATTTGGGTTAATCTAGCCAAATTGCGTGAAGAGATTTCAATTTGGTTGTGCTCATAAACCAAGTAGGAGTTTTATTAGCATCATAAAAATTGAAGCTACATATTAATTGGCACCCTCCATGGGAGTTGCTCGCATGCTACACTTCTCTACTGCTCTTAAAAGCTACTTATCATTACCTTACTTTTGGTTAGTCGTCCTCTGTTCTTTGTTTCTAACAGAAAAAGATGCTTTGCATACTTTTATTGTTACTCGCTTCATGTTTTTGCTCTAACAGAATGCCCATTTCCTCATATTTCTCTTCTGCCTTGTGTACATGTTAGAGACAATAATTTTGTGGTAAAATCCATTTCACTATATATATTCCACATGATGCATATGCTCCTTTCTAACCTTGATATACAAAAGAAAAGACTAATGGCGTTGTTAGCTATAGAAATATTTCAACATAAATTCATACTTTGCCTAGCTTTTTCTGTTTTTGTTAGTAGCTGCCAACTAGTTAGCGGTGATGAGTTGGCATTCTTTGCCAGGCTTAATCTTCCAACAGGGGCAAGGGGTCCAGAATCTCTTGCTTTCAACGGCAAGGGGGACGAGTTTTATACCGGTGTATCTGATAGTAGAATCCTCAAGTATGAGTTGGCGAATCATGCTTTCGTCAATTTTGCAACCACCTCACCTCTCAGGTAAATTTTGCATCAGTTGTGCTTTTCGCCTTTGCAATAAGGAGCTGTTTGAATAAGCTTATTTTTCAGAAATAAGAGCTTATATTTGGAGATCTATTTTTTCCTGAAATAAGCCCTTTTGTTTGTCAAATAACGAATATGAAACACTTGTTTAACATTTATATCCAAAAATATTATAAAAAGATGCATTTATTTTTATAAAAATAAGCCTTAAATTTGATAAGTATGATCTGGCAAACGGGTGCTAAGTTTGAAGGAGAAATTAGCTACTTATAATAATGATATTAGCCTTCTCCAGACGGACAGACACCATTTGAATGCATAGTGTAGCATAGTTGAAGATCAAGTTTTGGGGAATGATATTGAAATTATAATCACCCATCCATTTTCAGGAACAAAGCAATTGGTGATGACAGAAACGATACAAAATTGGGAGAAGCTCCTTGTAGGCCATTAGGAGTCGCAATCAACTACAATAATGGTGACCTTTATATTGCGGATGGTGTATGTGGACTAGCTGTAGTGGGTTCAAATGGAGGTGTTGCCAATACAATTGCCAACAGCGTTAATGGCATTCCTCTTCTGTTTCCCAATGCCATTGATATCGATCCAAAAACAGAAACTGTTTATTTTACAGATGCTGGCTCTATTTTTCAGAAAAGGTTGATCACACACATATGTTGCTTCTTTTTCCAATCAAGCTTATGCTCTATATTCATTGAGTTTATTCTGTTTGTCTTTGAAATGTTGCAGTATGGACATTATAGAAATACTCTCAAGTGGAGACACGAGTGGAAAATTGCTCAAATATGATCCACAAACCAAACAAATTTCAGTGTTGCTAACAGAACTTTCTGGTGCAACTGGTGTAGCGGTTAGTGGAGATGGAGCTTTTGTGCTTGTTTCGGAGTACATTGCCAAGCAAATCCGCAGATACTGGATTAGAGGACCGCTAGCCGGAACATCAGACATATTTATTGAACTTGCAGGGAGCCCTGATAACATCAAAAGGACCGTGTTGGGAGATTTCTGGGTTGCAGTGACCATGGTGGATTTGCAGCTAGCTGTACCAATACTGGTGCCTTTCGGACAAAGAATAAATCCTATTGGTGTAATTCTGGAGAACTTTAGTCTTGAGGTGCAATACAGAAATGCAATTGTTAGTGAGGTTCAAGAAAATGCATGGGGAATATTTGTTGGAACCCTTTTAGGGAGTTCTGTTGGAATTTACAGAAGATAGTAATAACAAATTTATAACAATAAAACATTAGAGAACA"

vector = generate_kmer_vector(seq)

model, label_encoder = load_model('model.pth')
predict_and_plot(model, vector, label_encoder)
print(predict_and_get_lists(model, vector, label_encoder))

