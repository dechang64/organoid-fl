import Zai from 'z-ai-web-dev-sdk';
import fs from 'fs';

async function main() {
  const prompt = process.argv[2];
  const imgPath = process.argv[3];
  
  const zai = await Zai.create();
  const imgBuf = fs.readFileSync(imgPath);
  const imgB64 = imgBuf.toString('base64');
  
  // Retry on rate limit
  for (let attempt = 0; attempt < 3; attempt++) {
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
      console.log(JSON.stringify(resp.choices[0].message.content));
      return;
    } catch (e) {
      if (e.message && e.message.includes('429') && attempt < 2) {
        await new Promise(r => setTimeout(r, 5000 * (attempt + 1)));
        continue;
      }
      console.error(JSON.stringify({error: e.message}));
      process.exit(1);
    }
  }
}
main().catch(e => { console.error(e.message); process.exit(1); });
