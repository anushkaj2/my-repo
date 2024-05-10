# This code finds the genes that are mapped to the probe IDs in the "labels.csv" file

# Install the library: illuminaHumanv4.db
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("illuminaHumanv4.db")


# Load the probeIDs file
probeID = read.csv(file = "C:/Users/bindu/OneDrive/Desktop/labels.csv", header = FALSE, skip=1)
print(head(probeID))
is.data.frame(probeID)

# Because probeID is not a vector, convert it into one
probeIDvec = unlist(probeID)
is.vector(probeIDvec)

# Use the library to perform gene to probe mapping
library("illuminaHumanv4.db")
probe2gene = data.frame(Gene=unlist(mget(x = probeIDvec,envir = illuminaHumanv4SYMBOL)))
print(head(probe2gene))

# Export the dataframe as a csv file
write.csv(probe2gene, file = "probe2gene.csv")
