## Load Gretel NER data into Elasticsearch

Elasticsearch and Kibana provide an industry standard platform for getting a quick start on exploring your data,
building dashboards and running complex queries to search for records of interest.  This blueprint
looks at using these tools to examine records enriched with Gretel NER labels.

In this blueprint we show how to use the Gretel python client to bootstrap a project
with sample data, load the NER results into Elasticsearch and inspect the results 
via programmatic queries.  

For this blueprint notebook you will need your Gretel API Key.  You can find this in the Gretel Console
under the Profile menu or at https://console.gretel.cloud/users/me/key.  You will also need to have
Elasticsearch and Kibana running on localhost.  The notebook sets up a local cluster with docker.

