To build a high-confidence, incremental face clustering system with FaceNet embeddings, the most robust approach is a **Semi-Supervised Incremental Hierarchical Clustering** (or a variation of **Constrained DBSCAN**).

FaceNet embeddings are normalized to a hypersphere ($L\_2$ distance), making cosine similarity and Euclidean distance highly effective. Because you need human overrides (positive/negative constraints), a graph-based or density-based approach is superior to K-Means.

---

## **1\. The Core Algorithm: Constrained Incremental DBSCAN**

DBSCAN is ideal for faces because it handles "noise" (outliers like blurry or side-profile faces) and doesn't require a pre-defined number of clusters.

### **The Strategy**

* **Initial Pass:** Run **HDBSCAN** or **DBSCAN** on your starting set. Use a conservative distance threshold (e.g., $d \< 0.6$ for FaceNet 128D) to ensure high precision.  
* **Incremental Step:** When a new face vector ($v\_{new}$) arrives, do not re-run the entire database. Instead, use a **Nearest Neighbor Search** (like FAISS or Annoy) to find existing clusters within your threshold.  
* **Confidence Scoring:** Calculate the "Core Distance" (mean distance to the $k$-nearest neighbors in a cluster). If $v\_{new}$ is significantly closer to the center than the cluster's boundary, it is auto-assigned. If it’s on the edge, it’s flagged for **Human Review**.

---

## **2\. Handling Human Overrides (Semi-Supervision)**

You can treat human input as **Must-Link** (Positive) and **Cannot-Link** (Negative) constraints. This transforms the problem into **Constrained Clustering**.

| Override Type | Implementation Detail |
| :---- | :---- |
| **Positive (Merge)** | Force an edge between two vectors in the similarity graph. In a density-based model, you can "artificially" set their distance to 0\. |
| **Negative (Split)** | Assign an "Infinite Distance" weight between two vectors or clusters. This prevents the algorithm from bridging them even if a new "middle-man" face appears. |
| **"Locking"** | Once a human verifies a cluster, mark its members as verified. The algorithm can add to it, but it cannot dissolve or merge it with another verified cluster without permission. |

---

## **3\. Step-by-Step Workflow**

### **Phase 1: Bootstrap**

1. **Pre-process:** Ensure all FaceNet vectors are $L\_2$ normalized.  
2. **Index:** Insert the small starting set into a vector database (FAISS).  
3. **Cluster:** Run **DBSCAN**. Assign Cluster IDs to each vector.

### **Phase 2: Incremental Growth**

1. **Query:** For a new vector $x$, query the FAISS index for the $K$ nearest neighbors ($k=5$ is usually sufficient).  
2. **Evaluate:**  
   * **Case A (High Confidence):** All $K$ neighbors belong to Cluster $A$ and distance $\<\\epsilon$. Auto-assign to $A$.  
   * **Case B (Ambiguous):** Neighbors belong to different clusters ($A$ and $B$). This is a potential **Merge Conflict**.  
   * **Case C (Outlier):** No neighbors within $\\epsilon$. Create a temporary "Unassigned" group.

### **Phase 3: Human-in-the-Loop (HITL)**

* **Conflict Resolution:** Present Case B (Ambiguous) to a human. If they say "Yes, they are the same," merge Cluster $A$ and $B$. If "No," add a **Cannot-Link** constraint.  
* **New Identity Discovery:** Once an "Unassigned" group reaches a certain size (e.g., 5-10 similar faces), prompt the user to name the new person.

---

## **4\. Technical Recommendations**

* **Distance Metric:** Use **Squared Euclidean Distance** ($d^2$) or **Cosine Similarity**. For FaceNet, the standard threshold for "same person" is typically between $0.4$ and $0.6$ (lower is stricter).  
* **Performance:** Use **FAISS** (Facebook AI Similarity Search). It allows you to add vectors incrementally to an index without rebuilding, supporting millions of faces with sub-millisecond lookups.  
* **Refinement:** Periodically (e.g., once a week), run a global **Graph-based Cleanup** (like the *Chinese Whispers*algorithm) to optimize cluster boundaries based on the accumulated human constraints.

---

## **5\. Implementation overview**

To implement this, we'll use **FAISS** for fast indexing and a **custom Incremental Graph** approach. Standard DBSCAN isn't natively incremental, so the best way to handle "High Confidence" clusters while allowing "Human Overrides" is to manage a cluster-map alongside a similarity index.

This script demonstrates a class that:

1. **Bootstraps** from a small set.  
2. **Adds** new vectors one by one.  
3. **Applies "Cannot-Link" constraints** (Human Negatives).  
4. **Auto-merges** only when confidence is high (Human Positives).

   import faiss

   import numpy as np

   

   class IncrementalFaceClustering:

       def \_\_init\_\_(self, dimension=128, threshold=0.6):

           \# IndexFlatL2 for high precision; use IndexIVFFlat for millions of vectors

           self.index \= faiss.IndexFlatL2(dimension)

           self.threshold \= threshold  \# FaceNet L2 distance threshold

           self.vector\_metadata \= \[\]   \# List of {'cluster\_id': int, 'manual': bool}

           self.next\_cluster\_id \= 0

           self.cannot\_link \= set()    \# Stores tuple pairs (vector\_idx\_1, vector\_idx\_2)

   

       def bootstrap(self, vectors):

           """Initial cluster setup for your starting set."""

           for v in vectors:

               self.add\_vector(v)

   

       def add\_vector(self, v, manual\_cluster\_id=None):

           """Add a vector with optional human override."""

           v \= v.reshape(1, \-1).astype('float32')

           new\_idx \= self.index.ntotal

           

           \# 1\. Check for manual override

           if manual\_cluster\_id is not None:

               self.index.add(v)

               self.vector\_metadata.append({'cluster\_id': manual\_cluster\_id, 'manual': True})

               return manual\_cluster\_id

   

           \# 2\. Search for neighbors within threshold

           \# We look for the 5 nearest to check for cluster consistency

           distances, indices \= self.index.search(v, 5\)

           

           valid\_indices \= indices\[0\]\[(distances\[0\] \< self.threshold) & (indices\[0\] \!= \-1)\]

           

           if len(valid\_indices) \== 0:

               \# No close matches \-\> New Identity

               cid \= self.next\_cluster\_id

               self.next\_cluster\_id \+= 1

           else:

               \# Check for Cannot-Link constraints with potential neighbors

               potential\_cids \= \[self.vector\_metadata\[i\]\['cluster\_id'\] for i in valid\_indices\]

               

               \# Simple logic: Assign to the most frequent nearby cluster 

               \# (In a production system, you'd check for constraint violations here)

               cid \= max(set(potential\_cids), key=potential\_cids.count)

   

           self.index.add(v)

           self.vector\_metadata.append({'cluster\_id': cid, 'manual': False})

           return cid

   

       def add\_negative\_constraint(self, idx1, idx2):

           """Human says: These two specific images are NOT the same person."""

           self.cannot\_link.add(tuple(sorted((idx1, idx2))))

           \# Logic to split clusters if they were previously joined would go here.

   

   \# \--- Usage Example \---

   \# 1\. Initialize (FaceNet produces 128-d or 512-d vectors)

   clf \= IncrementalFaceClustering(dimension=128, threshold=0.55)

   

   \# 2\. Add starting set

   start\_vectors \= np.random.random((10, 128)) 

   clf.bootstrap(start\_vectors)

   

   \# 3\. Add a new face found in the wild

   new\_face \= np.random.random((1, 128))

   assigned\_id \= clf.add\_vector(new\_face)

   print(f"Vector assigned to Cluster: {assigned\_id}")

   

   \# 4\. Human Override: "Actually, this is person \#5"

   clf.add\_vector(new\_face, manual\_cluster\_id=5)

### **Why this works for your needs:**

* **High Confidence:** By setting a strict threshold (e.g., $0.5$ for FaceNet), the system only auto-clusters faces that are mathematically very similar.  
* **Incremental:** Using index.add(), you only process the new vector against the existing index ($O(\\log N)$ or $O(1)$ search) rather than re-clustering the whole DB ($O(N^2)$).  
* **Human Overrides:** The manual\_cluster\_id parameter allows you to bypass the algorithm's logic. You can easily extend this to a "Must-Link" table that merges IDs globally.  
* **Memory Efficient:** FAISS handles the heavy lifting of vector math in C++, keeping the Python side light for your metadata and constraints.

**Would you like me to refine the "Cannot-Link" logic to show how to automatically trigger a cluster split when a negative constraint is added?**

