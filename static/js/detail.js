/* ==========================================================================
   详情页：图片切换 + 匹配结果重新计算
   ========================================================================== */
'use strict';

(function () {
  // ---- 图库切换 ----
  const mainImg = document.getElementById('gallery-main-img');
  document.querySelectorAll('.gallery-thumb').forEach((thumb) => {
    thumb.addEventListener('click', () => {
      if (mainImg) mainImg.src = thumb.dataset.full;
      document.querySelectorAll('.gallery-thumb').forEach((t) => t.classList.remove('active'));
      thumb.classList.add('active');
    });
  });

  // ---- 重新计算匹配 ----
  const refreshBtn = document.querySelector('[data-action="refresh-matches"]');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', async () => {
      const itemId = refreshBtn.dataset.itemId;
      const list = document.getElementById('match-list');
      const counterpart = refreshBtn.dataset.counterpart || '对侧记录';
      refreshBtn.disabled = true;
      refreshBtn.textContent = '计算中…';
      try {
        const data = await LF.api(`/api/items/${itemId}/matches?top_k=6`);
        if (list) {
          list.innerHTML = LF.renderMatches(
            data.matches,
            `<div class="empty-state small"><h3>暂无匹配</h3><p>在库内 ${LF.esc(counterpart)} 中未找到相似记录。可以补充更多标签或更换更清晰的照片后重试。</p></div>`
          );
        }
        LF.toast(`重新计算完成：命中 ${data.count} 条候选`, 'ok');
      } catch (err) {
        LF.toast(err.message, 'error');
      } finally {
        refreshBtn.disabled = false;
        refreshBtn.textContent = '重新计算';
      }
    });
  }
})();
