## Gretel NER service Helm deployment

### Requirements
- A Kubernetes Cluster (HPA enabled is optional)
- Helm3+
- Gretel Project API Key

### Installation

```
kubectl create namespace gretel
kubectl -n gretel create secret docker-registry gretel --docker-server=https://registry-1.docker.io --docker-username=greteluser --docker-password=<token> --docker-email=<your email>
helm install --namespace gretel gretel-ner . --set apiToken=grtuxyz
```

### Testing
```
kubectl -n gretel port-forward gretel-ner-server-7b6bc4785-5sd5r 8000:8000 > /dev/null &
sleep 2
curl --request POST \
  --url http://localhost:8000/records/detect_entities \
  --header 'content-type: application/json' \
  --data '{
	"email": "gretel@gmail.com",
	"ssn": "199833782"
}'
```
