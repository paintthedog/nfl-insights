// Quick pre-publish check (Oct 21; replaces the old scratchpad t2.js/nav.js): the page's data parses and its scripts compile.
// Usage: node check_page.js site/paydirt.html
const s = require('fs').readFileSync(process.argv[2] || 'site/paydirt.html', 'utf8');
let bad = 0;
for (const m of s.matchAll(/<script(?![^>]*src)[^>]*>([\s\S]*?)<\/script>/g)) { try { new Function(m[1]); } catch (e) { bad++; console.log('SCRIPT ERROR', e.message); } }
const i = s.indexOf('const ALL = ') + 12; let d = 0, j = i;
for (; j < s.length; j++) { if (s[j] === '{') d++; else if (s[j] === '}' && --d === 0) break; }
const A = JSON.parse(s.slice(i, j + 1));
const weeks = A.weeks.map(w => `wk${w}: ${A.data[w].proj.players.length} players, built ${A.data[w].built}`);
if (s.includes('my_bets')) { bad++; console.log('PRIVATE BET LOG REFERENCED'); }
console.log(weeks.join(' | '), '|', bad ? `${bad} PROBLEM(S)` : 'OK');
process.exit(bad ? 1 : 0);
