# Blueprint Template

Your blueprint directory **MUST** have the following items:

- `manifest.json`: See the example manifest for reference.
    - Your blueprint name must be unique
    - If your blueprint uses a different datasource other than sample data from the Gretel REST API, you can set that value to `null`
    - There are restrictions for the name and description sizes:
      - Name: 64 chars
      - Description: 160 chars
      - Tags: 5 tags max, each tag max 32 chars

- `blueprint.XXX`: This should be the actual blueprint, the file extension can vary based on the actual blueprint type. If you are using Python, this could be `.ipynb`, for JavaScript, `.js` etc. Regardless the base file name **MUST** be `blueprint`.