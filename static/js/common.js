/* ==========================================================================
   公共工具：API 封装、Toast、模板渲染、导航交互
   纯原生 JS（无构建步骤），所有页面共享。
   ========================================================================== */
'use strict';

const LF = (() => {
  /** 统一的后端接口调用：自动处理 JSON 与错误信息 */
  async function api(path, { method = 'GET', body, formData, headers } = {}) {
    const options = { method, headers: { ...(headers || {}) }, credentials: 'same-origin' };
    if (formData) {
      options.body = formData;                 // multipart：不要手动设置 Content-Type
    } else if (body !== undefined) {
      options.headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(body);
    }
    let response;
    try {
      response = await fetch(path, options);
    } catch (err) {
      throw new Error('网络请求失败，请检查服务是否在运行');
    }
    const text = await response.text();
    let data = {};
    if (text) {
      try { data = JSON.parse(text); } catch (err) { data = { ok: false, message: text.slice(0, 200) }; }
    }
    if (!response.ok || data.ok === false) {
      throw Object.assign(new Error(data.message || `请求失败（HTTP ${response.status}）`), {
        status: response.status,
        code: data.error || 'error',
        payload: data,
      });
    }
    return data;
  }

  /** 轻量提示条 */
  function toast(message, kind = 'info', timeout = 3200) {
    const host = document.getElementById('toast-host');
    if (!host) { console.log('[toast]', kind, message); return; }
    const el = document.createElement('div');
    el.className = `toast ${kind}`;
    el.textContent = message;
    host.appendChild(el);
    setTimeout(() => {
      el.style.transition = 'opacity .25s, transform .25s';
      el.style.opacity = '0';
      el.style.transform = 'translateY(6px)';
      setTimeout(() => el.remove(), 260);
    }, timeout);
  }

  const ESCAPE_MAP = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  /** HTML 转义：所有插入 DOM 的用户内容都必须经过它，避免 XSS */
  function esc(value) {
    return String(value === null || value === undefined ? '' : value).replace(/[&<>"']/g, (ch) => ESCAPE_MAP[ch]);
  }

  function pct(value) {
    const num = Number(value || 0);
    return `${Math.round(num * 100)}%`;
  }

  function scoreClass(score) {
    const s = Number(score || 0);
    if (s >= 0.55) return 'strong';
    if (s >= 0.42) return 'likely';
    if (s >= 0.30) return 'candidate';
    return 'weak';
  }

  function scoreLabel(score) {
    const s = Number(score || 0);
    if (s >= 0.55) return '高可信匹配';
    if (s >= 0.42) return '较可能匹配';
    if (s >= 0.30) return '疑似匹配';
    return '弱相关';
  }

  /** 读取图片文件为 dataURL，并做等比压缩（避免把原始大图塞进请求体） */
  function fileToDataURL(file, maxSide = 1280, quality = 0.86) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error('读取文件失败'));
      reader.onload = () => {
        const img = new Image();
        img.onerror = () => reject(new Error('无法解析图片'));
        img.onload = () => {
          const scale = Math.min(1, maxSide / Math.max(img.width, img.height));
          if (scale >= 1) { resolve(reader.result); return; }
          const canvas = document.createElement('canvas');
          canvas.width = Math.round(img.width * scale);
          canvas.height = Math.round(img.height * scale);
          canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
          resolve(canvas.toDataURL('image/jpeg', quality));
        };
        img.src = reader.result;
      };
      reader.readAsDataURL(file);
    });
  }

  /** 渲染单条匹配结果卡片（与后端 _match_card.html 结构保持一致） */
  function renderMatch(match, options = {}) {
    const item = match.item || {};
    const detail = match.detail || {};
    const weights = detail.weights || {};
    const img = detail.image || {};
    const txt = detail.text || {};
    const cover = (item.images && item.images[0]) ? item.images[0].thumb : null;
    const score = Number(match.score || 0);
    const reasons = (match.reasons || []).map((r) => `<li>${esc(r)}</li>`).join('');
    const locationBar = Number(match.location_score || 0) > 0
      ? `<div class="bar-row"><span>地点</span><div class="bar bar-3"><i style="width:${pct(match.location_score)}"></i></div><b>${pct(match.location_score)}</b></div>`
      : '';

    return `
      <article class="match-item" data-match-id="${esc(item.id)}">
        <div class="match-item-head">
          <a class="match-thumb" href="/item/${esc(item.id)}">
            ${cover
              ? `<img src="/static/uploads/${esc(cover)}" alt="${esc(item.title)}">`
              : `<span class="mini-noimg">无图</span>`}
          </a>
          <div class="match-item-info">
            <div class="match-score-row">
              <span class="score-chip ${scoreClass(score)}">${pct(score)}</span>
              <span class="level-label">${esc(match.level_label || scoreLabel(score))}</span>
            </div>
            <a class="match-item-title" href="/item/${esc(item.id)}">${esc(item.title)}</a>
            <p class="muted small">
              ${esc(item.type_label || '')} · ${esc(item.category_name || '未分类')}
              ${item.location ? ' · 📍 ' + esc(item.location) : ''}
            </p>
          </div>
        </div>

        <div class="score-bars">
          <div class="bar-row"><span>图像</span><div class="bar"><i style="width:${pct(match.image_score)}"></i></div><b>${pct(match.image_score)}</b></div>
          <div class="bar-row"><span>文本</span><div class="bar bar-2"><i style="width:${pct(match.text_score)}"></i></div><b>${pct(match.text_score)}</b></div>
          ${locationBar}
        </div>

        <ul class="reason-list">${reasons}</ul>

        <details class="match-detail">
          <summary>查看技术细节</summary>
          <table class="kv">
            <tr><th>融合权重</th><td>图像 ${Math.round((weights.image || 0) * 100)}% · 文本 ${Math.round((weights.text || 0) * 100)}% · 地点 ${Math.round((weights.location || 0) * 100)}%</td></tr>
            <tr><th>轮廓相似度 (Dice)</th><td>${esc(img.shape !== undefined ? img.shape : '—')}</td></tr>
            <tr><th>颜色直方图相交</th><td>${esc(img.color !== undefined ? img.color : '—')}</td></tr>
            <tr><th>主色对比</th><td>${esc((img.dominant_labels || []).filter(Boolean).join(' | ') || '—')}</td></tr>
            <tr><th>整图哈希距离</th><td>${img.hamming !== undefined ? esc(img.hamming) + ' / 64（仅作同图提示）' : '—'}</td></tr>
            <tr><th>TF-IDF 余弦</th><td>${esc(txt.tfidf !== undefined ? txt.tfidf : '—')}</td></tr>
            <tr><th>标签重合度</th><td>${esc(txt.tag !== undefined ? txt.tag : '—')}</td></tr>
          </table>
        </details>

        <div class="match-item-foot">
          <a class="btn btn-sm btn-outline" href="/item/${esc(item.id)}">查看详情</a>
          ${item.contact ? `<span class="contact-inline">联系方式：${esc(item.contact)}</span>` : ''}
        </div>
      </article>`;
  }

  function renderMatches(matches, emptyHtml) {
    if (!matches || !matches.length) {
      return emptyHtml || `<div class="empty-state small"><h3>暂无匹配</h3><p>库内还没有相似的对侧记录，可以补充标签后重试。</p></div>`;
    }
    return matches.map((m) => renderMatch(m)).join('');
  }

  /** 退出登录 */
  async function logout() {
    try {
      await api('/api/auth/logout', { method: 'POST' });
      toast('已退出登录', 'ok');
      setTimeout(() => window.location.reload(), 400);
    } catch (err) {
      toast(err.message, 'error');
    }
  }

  function init() {
    // 用户菜单
    document.querySelectorAll('[data-user-menu]').forEach((wrap) => {
      const btn = wrap.querySelector('.user-chip-btn');
      const menu = wrap.querySelector('.user-menu');
      if (!btn || !menu) return;
      btn.addEventListener('click', (event) => {
        event.stopPropagation();
        const open = !menu.hidden;
        menu.hidden = open;
        btn.setAttribute('aria-expanded', String(!open));
      });
      document.addEventListener('click', () => { menu.hidden = true; });
    });

    // 退出按钮（头部菜单或页面内任意位置）
    document.addEventListener('click', (event) => {
      const trigger = event.target.closest('[data-action="logout"]');
      if (trigger) logout();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  return { api, toast, esc, pct, scoreClass, scoreLabel, fileToDataURL, renderMatch, renderMatches, logout };
})();
