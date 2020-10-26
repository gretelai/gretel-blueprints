# Gretel Blueprints

Gretel Blueprints provide example code for solving specific use cases. Gretel managed use cases can be located in the 
`gretel` directory. Each Blueprint has its own directory. If the directory is prefixed with `gc-` then that Blueprint is 
optimized to download labeled records from a Gretel Cloud Project. Blueprints without this directory prefix allow you to
bring your own datasets from anywhere.

Blueprint contents and any supporting modules within the Blueprint sub-directories are licensed under Apache 2.0.  Most Blueprints also will install our [gretel-client](https://github.com/gretelai/gretel-python-client) and [gretel-synthetics](https://github.com/gretelai/gretel-synthetics) open source packages.

However, Blueprints will also download and install the Gretel Premium Packages and the assoicated COPYRIGHT or Gretel License associated with these installed packages still applies separately from the Apache 2.0 licensed code.  Currently, these packages are installed via our `gretel-client` package.

If you have any questions, comments, or bugs regarding the blueprints please feel free to open a GitHub issue or reach out to us at hi@gretel.ai!
