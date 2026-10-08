/* ==========================================================================
   算法演示台：双通道得分实时试算（含可选的临时图片比对）
   ========================================================================== */
'use strict';

(function () {
  const runBtn = document.getElementById('run-btn');
  if (!runBtn) return;

  const resultBody = document.getElementById('algo-result-body');

  const sliders = {
    image: document.getElementById('w-image'),
    text: document.getElementById('w-text'),
    location: document.getElementById('w-location'),
  };
  const sliderLabels = {
    image: document.getElementById('w-image-val'),
    text: document.getElementById('w-text-val'),
    location: document.getElementById('w-location-val'),
  };

  /** 各侧已选图片的 dataURL */
  const images = { a: [], b: [] };

  function bindSlider(key) {
    if (!sliders[key]) return;
    const sync = () => { if (sliderLabels[key]) sliderLabels[key].textContent = `${sliders[key].value}%`; };
    sliders[key].addEventListener('input', sync);
    sync();
  }
  Object.keys(sliders).forEach(bindSlider);

  function bindImageInput(side) {
    const input = document.getElementById(`${side}-images`);
    const preview = document.getElementById(`${side}-preview`);
    if (!input || !preview) return;
    input.addEventListener('change', async () => {
      const picked = Array.from(input.files || []).slice(0, 2);
      images[side] = [];
      preview.innerHTML = '';
      for (const file of picked) {
        try {
          const url = await LF.fileToDataURL(file, 1024, 0.85);
          images[side].push(url);
          const wrap = document.createElement('div');
          wrap.className = 'preview-item';
          wrap.innerHTML = `<img src="${url}" alt="预览">`;
          preview.appendChild(wrap);
        } catch (err) {
          LF.toast(err.message, 'error');
        }
      }
    });
  }
  bindImageInput('a');
  bindImageInput('b');

  function readSide(prefix) {
    return {
      title: document.getElementById(`${prefix}-title`).value.trim(),
      tags: document.getElementById(`${prefix}-tags`).value.split(/[,，、]/).map((t) => t.trim()).filter(Boolean),
      color_name: document.getElementById(`${prefix}-color`).value.trim(),
      location: document.getElementById(`${prefix}-location`).value.trim(),
      description: document.getElementById(`${prefix}-desc`).value.trim(),
    };
  }

  function renderResult(payload, totalScore, level, label) {
    const d = payload.detail || {};
    const weights = d.weights || {};
    const img = d.image || {};
    const txt = d.text || {};
    const reasons = (payload.reasons || []).map((r) => `<li>${LF.esc(r)}</li>`).join('');

    resultBody.innerHTML = `
      <div class="algo-verdict">
        <span class="score-big ${LF.scoreClass(totalScore)}">${LF.pct(totalScore)}</span>
        <span>
          <strong>${LF.esc(label)}</strong><br>
          <small class="muted">图像 ${LF.pct(payload.image_score)} · 文本 ${LF.pct(payload.text_score)} · 地点 ${LF.pct(payload.location_score)}</small>
        </span>
      </div>

      <div class="score-bars">
        <div class="bar-row"><span>图像</span><div class="bar"><i style="width:${LF.pct(payload.image_score)}"></i></div><b>${LF.pct(payload.image_score)}</b></div>
        <div class="bar-row"><span>文本</span><div class="bar bar-2"><i style="width:${LF.pct(payload.text_score)}"></i></div><b>${LF.pct(payload.text_score)}</b></div>
        <div class="bar-row"><span>地点</span><div class="bar bar-3"><i style="width:${LF.pct(payload.location_score)}"></i></div><b>${LF.pct(payload.location_score)}</b></div>
      </div>

      <h3 style="margin-top:1rem">匹配理由</h3>
      <ul class="reason-list">${reasons}</ul>

      <h3>通道分解</h3>
      <table class="kv">
        <tr><th>归一化权重</th><td>图像 ${Math.round((weights.image || 0) * 100)}% · 文本 ${Math.round((weights.text || 0) * 100)}% · 地点 ${Math.round((weights.location || 0) * 100)}%</td></tr>
        <tr><th>轮廓相似度 (Dice)</th><td>${img.shape !== undefined ? img.shape : '—'}</td></tr>
        <tr><th>颜色直方图相交</th><td>${img.color !== undefined ? img.color : '—'}（前景 ${img.crop_color !== undefined ? img.crop_color : '—'}）</td></tr>
        <tr><th>主色一致性</th><td>${img.dominant_sim !== null && img.dominant_sim !== undefined ? img.dominant_sim : '—'}</td></tr>
        <tr><th>主色对比</th><td>${LF.esc((img.dominant_labels || ['', '']).filter(Boolean).join(' | ') || '—')}</td></tr>
        <tr><th>整图哈希距离</th><td>${img.hamming !== undefined ? img.hamming + ' / 64（仅作同图提示）' : '—'}</td></tr>
        <tr><th>TF-IDF 余弦</th><td>${txt.tfidf !== undefined ? txt.tfidf : '—'}</td></tr>
        <tr><th>标签重合度</th><td>${txt.tag !== undefined ? txt.tag : '—'}</td></tr>
        <tr><th>token 重合度</th><td>${txt.token !== undefined ? txt.token : '—'}</td></tr>
        <tr><th>命中标签</th><td>${LF.esc((txt.tag_hits || []).join('、') || '—')}</td></tr>
      </table>`;
  }

  async function run() {
    const left = readSide('a');
    const right = readSide('b');
    const weights = {
      image: Number(sliders.image.value) / 100,
      text: Number(sliders.text.value) / 100,
      location: Number(sliders.location.value) / 100,
    };

    runBtn.disabled = true;
    runBtn.textContent = '比对中…';

    try {
      const data = await LF.api('/api/match/pair', {
        method: 'POST',
        body: {
          left: { ...left, images: images.a },
          right: { ...right, images: images.b },
          weights,
        },
      });
      const payload = data.result;
      renderResult(payload, payload.score, payload.level, payload.level_label);
      if (images.a.length || images.b.length) {
        LF.toast(`已纳入图像通道（A ${images.a.length} 张 / B ${images.b.length} 张）`, 'ok', 2200);
      }
    } catch (err) {
      resultBody.innerHTML = `<p class="muted">比对失败：${LF.esc(err.message)}</p>`;
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = '开始比对';
    }
  }

  runBtn.addEventListener('click', run);
  // 首次进入自动跑一次，页面不空着
  run();
})();
