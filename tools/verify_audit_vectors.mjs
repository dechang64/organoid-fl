// ── tools/verify_audit_vectors.mjs ──
// Independent cross-check of the canonical audit preimage using Node's own
// JSON encoder and crypto, so the spec is validated outside Python's json
// module. Run:  node tools/verify_audit_vectors.mjs
//
// This mirrors what the Rust implementation must do: sort object keys, emit
// compact JSON, hash the UTF-8 bytes with SHA-256.
//
// Third leg of G-1 (organoid-audit-v1 in Python / Rust / JavaScript). The MVP's
// in-page chain was FNV-1a, which no server-side verifier could reproduce; this
// script is the evidence that the spec is implementable in the browser's
// language with byte-identical output.
//
// The float guard is not decoration. `JSON.stringify` prints 1e-5 as "0.00001"
// and 1e-7 as "1e-7"; Python's repr and Rust's Display switch at different
// points. Rather than emit a digest that only Node can read back, numbers
// outside [1e-4, 1e15) are refused — the same domain `audit_spec.validate_float`
// enforces. Refusing is what keeps the three implementations comparable.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));

const FLOAT_MIN_ABS = 1e-4;
const FLOAT_MAX_ABS = 1e15;

class CanonicalError extends Error {}

/** Compact JSON with lexicographically sorted object keys. */
function canonicalStringify(value, path = "$") {
  if (value === null || typeof value !== "object") {
    if (typeof value === "number" && !Number.isInteger(value)) {
      const magnitude = Math.abs(value);
      if (magnitude !== 0 && (magnitude < FLOAT_MIN_ABS || magnitude >= FLOAT_MAX_ABS)) {
        throw new CanonicalError(
          `${path} = ${value} is outside the canonical float domain ` +
            `[${FLOAT_MIN_ABS}, ${FLOAT_MAX_ABS}); Python, Rust and JavaScript ` +
            "disagree about exponent notation there, so the digest would depend " +
            "on the writing language. Store basis points or micrometres as integers.",
        );
      }
    }
    if (typeof value === "number" && !Number.isFinite(value)) {
      throw new CanonicalError(`${path} is ${value}; it has no canonical JSON form`);
    }
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return "[" + value.map((item, i) => canonicalStringify(item, `${path}[${i}]`)).join(",") + "]";
  }
  const keys = Object.keys(value).sort();
  return (
    "{" +
    keys
      .map((k) => JSON.stringify(k) + ":" + canonicalStringify(value[k], `${path}.${k}`))
      .join(",") +
    "}"
  );
}

function canonicalHash({ index, timestamp, operation, details, prev_hash, nonce }) {
  const payload = {
    details: details ?? {},
    index,
    nonce: nonce ?? 0,
    operation,
    prev_hash,
    timestamp,
  };
  return createHash("sha256").update(canonicalStringify(payload), "utf8").digest("hex");
}

const fixturePath = join(here, "..", "tests", "fixtures", "audit_hash_vectors.json");
const fixture = JSON.parse(readFileSync(fixturePath, "utf8"));

let failures = 0;
for (const v of fixture.vectors) {
  const preimage = canonicalStringify({
    details: v.details ?? {},
    index: v.index,
    nonce: v.nonce ?? 0,
    operation: v.operation,
    prev_hash: v.prev_hash,
    timestamp: v.timestamp,
  });
  const digest = canonicalHash(v);

  const preimageOk = preimage === v.preimage;
  const digestOk = digest === v.sha256;
  if (!preimageOk || !digestOk) failures++;

  const mark = preimageOk && digestOk ? "OK  " : "FAIL";
  console.log(`${mark} ${v.name.padEnd(34)} ${digest}`);
  if (!preimageOk) {
    console.log(`     expected preimage: ${v.preimage}`);
    console.log(`     node     preimage: ${preimage}`);
  }
  if (!digestOk) {
    console.log(`     expected hash: ${v.sha256}`);
  }
}

console.log(
  failures === 0
    ? `\nall ${fixture.vectors.length} vectors reproduced under Node (spec ${fixture.spec_version})`
    : `\n${failures} of ${fixture.vectors.length} vectors MISMATCHED`,
);
process.exit(failures === 0 ? 0 : 1);
