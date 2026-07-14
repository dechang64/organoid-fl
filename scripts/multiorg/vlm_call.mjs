import Zai from 'z-ai-web-dev-sdk';
import fs from 'fs';
import path from 'path';

async function main() {
  const prompt = process.argv[2];
  const imgPath = process.argv[3];

  const zai = await Zai.create();
  const imgBuf = fs.readFileSync(imgPath);
  const imgB64 = imgBuf.toString('base64');

  // Retry on rate limit (429)
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const resp = await zai.chat.completions.create({
        model: 'glm-4v-plus',
        messages: [{
          role: 'user',
          content: [
            { type: 'image_url', image_url: { url: `data:image/png;base64,${imgB64}` }},
            { type: 'text', text: prompt }
          ]
        }]
      });
      // Output: just the content string, as JSON
      console.log(JSON.stringify(resp.choices[0].message.content));
      return;
    } catch (e) {
      const msg = e.message || '';
      if (msg.includes('429') && attempt < 4) {
        // Wait 3s, 6s, 9s, 12s before retry
        const wait = 3000 * (attempt + 1);
        process.stderr.write(`Rate limited, waiting ${wait/1000}s...\n`);
        await new Promise(r => setTimeout(r, wait));
        continue;
      }
      // Final failure: output error JSON
      console.log(JSON.stringify({error: msg}));
      process.exit(0);  // exit 0 so Python doesn't crash
    }
  }
}
main().catch(e => { console.log(JSON.stringify({error: e.message})); process.exit(0); });

