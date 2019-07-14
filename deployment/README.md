# Deploying This Microservice: GitHub Issue Featurizer

This directory contains manifests for the backend of the microservice that returns embeddings given an issue label and body.  This backend is associated with associated with gh-issue-labeler.com/text.

This is currently running on a GKE cluster.


## GCP Project Name??

There is a dedicated instance running in

* **GCP project**: ??
* **cluster**: ??
* **namespace**: mlapp

Deploying it

1. Create the deployment

   ```
   kubectl apply -f deployments.yaml  
   ```

1. Create the secret

   ```
   gsutil cp gs://github-probots_secrets/ml-app-inference-secret.yaml /tmp
   kubectl apply -f /tmp/ml-app-inference-secret.yaml
   ```

1. Create the ingress

   ```
   kubectl apply -f ingress.yaml
   ```
