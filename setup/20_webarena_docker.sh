#!/usr/bin/env bash

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Step 2 — stand up the 4 WebArena site containers, bake localhost base-URLs, commit the pristine images
# that wa_reset.py recreates per task, then generate the task configs + eval auth-states into fetched/.
# HEAVY: downloads ~tens of GB of Docker images the first time. Requires step 1 (primary env) done first.
#
# Image mirror (CMU). If you already have the tarballs locally, drop them in $TARBALLS and this skips the download.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
. "$ROOT/env.sh"
MIRROR="http://metis.lti.cs.cmu.edu/webarena-images"
TARBALLS="$SGE_FETCHED/wa_tarballs"; mkdir -p "$TARBALLS" "$SGE_FETCHED/webarena/.auth"
WA="$SGE_FETCHED/webarena_official"; WAPY="$SGE_VENV"   # merged: the harness lives in the primary env now

load () {  # load <tarname> <image-check>
  docker image inspect "$2" >/dev/null 2>&1 && { echo "  image $2 present"; return; }
  echo "  fetching $1 (large — tens of GB total across the 4 images; RESUMABLE, safe to Ctrl-C + re-run) ..."
  ( cd "$TARBALLS" && wget -c --progress=dot:giga "$MIRROR/$1" )   # -c resumes a partial file; shows progress
  echo "  docker load $1 ..."; docker load --input "$TARBALLS/$1"
}
run () { docker rm -f "$1" >/dev/null 2>&1 || true; docker run -d --name "$@"; }

echo "### load images ###"
load shopping_final_0712.tar               shopping_final_0712
load shopping_admin_final_0719.tar         shopping_admin_final_0719
load gitlab-populated-final-port8023.tar   gitlab-populated-final-port8023
load postmill-populated-exposed-withimg.tar postmill-populated-exposed-withimg

echo "### stand up + configure (localhost base-urls) ###"
run webarena-shopping        -p 7770:80   shopping_final_0712
run webarena-shopping-admin  -p 7780:80   shopping_admin_final_0719
run webarena-verified-gitlab -p 8023:8023 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start
run webarena-verified-reddit -p 9999:80   postmill-populated-exposed-withimg
echo "  waiting for services to boot (~60s; gitlab needs several minutes) ..."; sleep 60

# Magento base-urls -> localhost
docker exec webarena-shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:7770"
docker exec webarena-shopping mysql -u magentouser -pMyPassword magentodb -e \
  'UPDATE core_config_data SET value="http://localhost:7770/" WHERE path = "web/secure/base_url";'
docker exec webarena-shopping /var/www/magento2/bin/magento cache:flush
docker exec webarena-shopping-admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:7780"
docker exec webarena-shopping-admin mysql -u magentouser -pMyPassword magentodb -e \
  'UPDATE core_config_data SET value="http://localhost:7780/" WHERE path = "web/secure/base_url";'
docker exec webarena-shopping-admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
docker exec webarena-shopping-admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0
docker exec webarena-shopping-admin /var/www/magento2/bin/magento cache:flush
# GitLab external_url -> localhost (give it a few minutes first)
echo "  gitlab reconfigure (slow) ..."; sleep 180
docker exec webarena-verified-gitlab sed -i "s|^external_url.*|external_url 'http://localhost:8023'|" /etc/gitlab/gitlab.rb
docker exec webarena-verified-gitlab gitlab-ctl reconfigure

echo "### commit pristine *_configured images (the names wa_reset.py recreates from) ###"
docker commit webarena-shopping        shopping_configured
docker commit webarena-shopping-admin  shopping_admin_configured
docker commit webarena-verified-gitlab gitlab_configured
# reddit uses the raw image name directly (no config needed) — nothing to commit.

echo "### generate task configs + eval auth-states (into fetched/) ###"
export SHOPPING="http://localhost:7770" SHOPPING_ADMIN="http://localhost:7780/admin" \
       REDDIT="http://localhost:9999" GITLAB="http://localhost:8023" \
       WIKIPEDIA="http://localhost:8888" HOMEPAGE="http://localhost:4399" MAP="http://localhost:3000"
( cd "$WA" && "$WAPY" scripts/generate_test_data.py || true )         # -> config_files/*.json
mkdir -p "$WA/.auth"
# auth-states PER SITE (--site_list): the no-arg auto_login main() fans renew_comb across a ThreadPoolExecutor
# and never .result()s those futures, so any login failure is silently swallowed and it writes ZERO state files
# (exit 0). The per-site path runs renew_comb directly, so it actually writes the file (and surfaces errors).
for site in reddit shopping shopping_admin gitlab; do
  ( cd "$WA" && PLAYWRIGHT_BROWSERS_PATH="$PLAYWRIGHT_BROWSERS_PATH" "$WAPY" -m browser_env.auto_login \
      --site_list "$site" --auth_folder "$WA/.auth" ) || echo "  [warn] auto_login $site failed"
done
# arrange auth-states into the layout paths.py expects ($SGE_FETCHED/webarena/…)
S="$SGE_FETCHED/webarena"
cp -f "$WA/.auth/reddit_state.json"         "$S/reddit_state.json"          2>/dev/null || true
cp -f "$WA/.auth/shopping_state.json"       "$S/.auth/shopping_state.json"  2>/dev/null || true
cp -f "$WA/.auth/shopping_admin_state.json" "$S/shopping_admin_state.json"  2>/dev/null || true
cp -f "$WA/.auth/gitlab_state.json"         "$S/gitlab_state.json"          2>/dev/null || true
echo "WebArena backends ready: images committed, config_files generated, auth-states in $S"
