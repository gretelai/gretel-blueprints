const fs = require("fs");
const { parse } = require("csv-parse");

/**
 *
 * For each file in the sample data folder,
 * generate a fresh "preview" version that is the first 20 records of that dataset.
 * This allows us to serve these previews on Console without needing to load the
 * entire sample dataset.
 *
 */

const PREVIEW_SIZE = 21; // 20 data rows + header
const READ_DIR = "sample_data";
const PREVIEW_DIR = "sample_data_previews";

function createPreviewForFile(fileName) {
  const readStream = fs.createReadStream(`${READ_DIR}/${fileName}`);

  readStream.pipe(
    parse({}, (err, records) => {
      const previewRecords = records.slice(0, PREVIEW_SIZE);
      fs.writeFile(
        `${PREVIEW_DIR}/${fileName}`,
        previewRecords.join("\n"),
        "utf8",
        () => {
          console.log(`New preview file written: ${fileName}.`);
        }
      );
    })
  );
}

function generateSampleDataPreviews() {
  fs.readdir(READ_DIR, (err, files) => {
    if (err) {
      console.log("ERROR reading sample data files", err);
    }
    files.forEach((fileName) => {
      console.log("processing file", fileName);
      createPreviewForFile(fileName);
    });
  });
}

function createPreviewDir() {
  // this method prevents an error if the folder already exists
  fs.mkdirSync(PREVIEW_DIR, { recursive: true });
}

function execute() {
  createPreviewDir();
  generateSampleDataPreviews();
}

execute();
