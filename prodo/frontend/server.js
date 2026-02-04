const { createServer } = require("https");
const { parse } = require("url");
const next = require("next");
const fs = require("fs");
const path = require("path");

const hostname = "0.0.0.0";
const port = parseInt(process.env.PORT || "3100", 10);

const certsDir = path.join(__dirname, "..", "certs");
const httpsOptions = {
  key: fs.readFileSync(path.join(certsDir, "lan.key")),
  cert: fs.readFileSync(path.join(certsDir, "lan.crt")),
};

const app = next({ dev: false, hostname, port });
const handle = app.getRequestHandler();

app.prepare().then(() => {
  createServer(httpsOptions, (req, res) => {
    const parsedUrl = parse(req.url, true);
    handle(req, res, parsedUrl);
  }).listen(port, hostname, () => {
    console.log(`> CommandCenter HTTPS ready on https://${hostname}:${port}`);
  });
});
