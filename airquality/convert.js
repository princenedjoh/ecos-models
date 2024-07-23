const data = require('./airquality.json')
const { Parser } = require('json2csv');
const fs = require('fs');

const compiled = data.list.map((item) => {
  return {
    date: new Date(item.dt * 1000),
    aqi: item.main.aqi,
    ...item.components
  };
});

const fields = Object.keys(compiled[0]);
const opts = { fields };

try {
  const parser = new Parser(opts);
  const csv = parser.parse(compiled);
  fs.writeFileSync('output.csv', csv);
  console.log('CSV file has been saved.');
} catch (err) {
  console.error(err);
}