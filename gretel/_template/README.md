# Blueprint Template

Your blueprint directory **MUST** have the following items:

- `manifest.json`: See the example manifest for reference.
    - Your blueprint name must be unique
    - If your blueprint uses a different datasource other than sample data from the Gretel REST API, you can set that value to `null`

- `blueprint.XXX`: This should be the actual blueprint, the file extension can vary based on the actual blueprint type. If you are using Python, this could be `.ipynb`, for JavaScript, `.js` etc. Regardless the base file name **MUST** be `blueprint`.