# E1-5 失败四分类抽检样本

来源 `eval/results_confirm_1500/hybrid_rr-on_rw-off_seed42.jsonl`，只含失败样本（success 判定不通过）。
每类取前 5 条（按文件顺序，非刻意挑选）。

**核对方法**：先只看「原始事实」区块自行判类，再看末尾的脚本判定。

## 未召回（该类失败共 1 条，抽 1 条）

### 未召回-1　qid `5ab2b7405542991669774069`

原始事实：

- 问题：Have James Schuyler and H. G. Wells both published books?
- 标准答案：`yes`
- 模型答案：`Yes`
- 答案 token-F1 = 1.000
- gold 段落 2 条：[3, 32]
- 检索到（并集，共 20 条）：[0, 2, 6, 7, 10, 11, 12, 16, 17, 20, 22, 23, 27, 28, 29, 31, 34, 36, 37, 38]
- 进入上下文的 5 条：[20, 31, 34, 0, 36]
- 上下文里命中的 gold：[]　漏掉的 gold：[3, 32]
- 模型引用 [1, 2, 3, 4] → 映射到 pid [20, 31, 34, 0]
- 引用精确率 cit_p = 0.000，召回 cit_r = 0.000
- 终止原因 `answer`，步数 1

上下文原文（★ = gold）：

　[1] (The Story of a Great Schoolmaster) But in December 1922, after disagreements emerged with Sanderson's widow about his approach to the subject, Wells withdrew from the official biography (published in 1923 as "Sanderson of Oundle"; Wells wrote much of the text but the volume was published without listing an author) and published his own work separately.
　[2] (James Schuyler) James Marcus Schuyler (November 9, 1923 – April 12, 1991) was an American poet.
　[3] (First and Last Things) First and Last Things is a 1908 work of philosophy by H. G. Wells setting forth his beliefs in four "books" entitled "Metaphysics," "Of Belief," "Of General Conduct," and "Some Personal Things."
　[4] (H. G. Wells) Herbert George Wells (21 September 1866 – 13 August 1946), usually referred to as H. G. Wells, was an English writer.
　[5] (First and Last Things) Wells revised the book extensively in 1917, in response to his religious conversion, but later published a further revision in 1929 that restored much of the book to its earlier form.

<details><summary>脚本判定</summary>

`未召回`

</details>

## 召回未读入（该类失败共 34 条，抽 5 条）

### 召回未读入-1　qid `5ae6363b55429929b0807af0`

原始事实：

- 问题：When was the defending titlist of 2009–10 Biathlon World Cup – Pursuit Men born? 
- 标准答案：`27 January 1974`
- 模型答案：`Insufficient evidence`
- 答案 token-F1 = 0.000
- gold 段落 2 条：[6, 22]
- 检索到（并集，共 22 条）：[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22]
- 进入上下文的 5 条：[21, 15, 13, 19, 2]
- 上下文里命中的 gold：[]　漏掉的 gold：[6, 22]
- 模型引用 [1, 2] → 映射到 pid [21, 15]
- 引用精确率 cit_p = 0.000，召回 cit_r = 0.000
- 终止原因 `no_new_evidence`，步数 5

上下文原文（★ = gold）：

　[1] (2009–10 Biathlon World Cup – Pursuit Men) The 2009–10 Biathlon World Cup – Pursuit Men started on December 13, 2009 in Hochfilzen and finished on March 20, 2010 in Oslo.
　[2] (2008–09 Biathlon World Cup – Pursuit Men) The 2008-09 Biathlon World Cup/Pursuit Men started at Sunday December 7, 2008 in Östersund and will finish Saturday March 28, 2009 in Khanty-Mansiysk.
　[3] (2010–11 Biathlon World Cup – Pursuit Men) The 2010–11 Biathlon World Cup – Pursuit Men will start at Sunday December 5, 2009 in Östersund and will finish Saturday March 19, 2011 in Oslo.
　[4] (2009–10 Biathlon World Cup – Relay Men) The 2009–10 Biathlon World Cup – Relay Men will start at Sunday December 6, 2009 in Östersund and will finish Friday February 26, 2009 in Vancouver at the olympic Biathlon event.
　[5] (2009–10 Biathlon World Cup – Pursuit Women) The 2009–10 Biathlon World Cup – Pursuit Women started on December 13, 2009 in Hochfilzen and finished on March 20, 2010 in Oslo.

<details><summary>脚本判定</summary>

`召回未读入`

</details>

### 召回未读入-2　qid `5a8ce15d554299653c1aa12f`

原始事实：

- 问题：did Robert Wise have more award nominations than Zoltan Korda?
- 标准答案：`yes`
- 模型答案：`Insufficient evidence`
- 答案 token-F1 = 0.000
- gold 段落 3 条：[2, 27, 28]
- 检索到（并集，共 28 条）：[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 24, 25, 26, 27, 28, 29, 30]
- 进入上下文的 5 条：[14, 0, 26, 16, 29]
- 上下文里命中的 gold：[]　漏掉的 gold：[2, 27, 28]
- 模型引用 [1, 2, 3] → 映射到 pid [14, 0, 26]
- 引用精确率 cit_p = 0.000，召回 cit_r = 0.000
- 终止原因 `no_new_evidence`，步数 6

上下文原文（★ = gold）：

　[1] (Elephant Boy (film)) Documentary filmmaker Robert J. Flaherty, who produced some of the Indian footage, and supervising director Zoltan Korda, who completed the film, won the Best Director Award at the Venice Film Festival.
　[2] (Zoltan Korda) Zoltan Korda (June 3, 1895 – October 13, 1961) was a Hungarian-born motion picture screenwriter, director and producer.
　[3] (Robert Wise) Robert Earl Wise (September 10, 1914 – September 14, 2005) was an American film director, producer and editor.
　[4] (The Four Feathers (1939 film)) The Four Feathers is a 1939 British Technicolor adventure film directed by Zoltan Korda, starring John Clements, Ralph Richardson, June Duprez, and C. Aubrey Smith.
　[5] (Cash (1933 film)) Cash is a 1933 British comedy film directed by Zoltan Korda and starring Edmund Gwenn, Wendy Barrie and Robert Donat.

<details><summary>脚本判定</summary>

`召回未读入`

</details>

### 召回未读入-3　qid `5ae7940c55429952e35ea984`

原始事实：

- 问题：What is the name of shipping magnate Mærsk Mc-Kinney Møller' father?
- 标准答案：`A.P. Møller`
- 模型答案：`Arnold Peter Møller`
- 答案 token-F1 = 0.400
- gold 段落 2 条：[14, 22]
- 检索到（并集，共 20 条）：[0, 1, 4, 5, 6, 7, 9, 11, 12, 13, 14, 16, 19, 21, 22, 25, 27, 28, 29, 30]
- 进入上下文的 5 条：[30, 13, 6, 12, 11]
- 上下文里命中的 gold：[]　漏掉的 gold：[14, 22]
- 模型引用 [1, 4] → 映射到 pid [30, 12]
- 引用精确率 cit_p = 0.000，召回 cit_r = 0.000
- 终止原因 `answer`，步数 1

上下文原文（★ = gold）：

　[1] (Peter Mærsk Møller) Peter Mærsk Møller (22 September 1836 – 9 February 1927) was a sea captain who perhaps is best known as the father of Arnold Peter Møller, founder of the Maersk corporation, and grandfather of Mærsk Mc-Kinney Møller, who made Maersk the largest container ship operator and supply vessel operator in the world.
　[2] (Mærsk Mc-Kinney Møller) Arnold Mærsk Mc-Kinney Møller (13 July 1913 – 16 April 2012) was a Danish shipping magnate.
　[3] (Ane Mærsk Mc-Kinney Uggla) Ane Mærsk Mc-Kinney Uggla (born 3 July 1948), often referred to as Ane Uggla, is a Swedish-Danish business executive and the youngest daughter of Mærsk Mc-Kinney Møller.
　[4] (Arnold Peder Møller) And he became the father of Mærsk Mc-Kinney Møller.
　[5] (Arnold Peder Møller) His father was Sea Captain Peter Mærsk Møller.

<details><summary>脚本判定</summary>

`召回未读入`

</details>

### 召回未读入-4　qid `5ae64ad25542995703ce8b4f`

原始事实：

- 问题：What was the name of the commander who captured all of the French ships that varied in size?
- 标准答案：`Joshua Rowley`
- 模型答案：`Joseph de Richery`
- 答案 token-F1 = 0.000
- gold 段落 3 条：[16, 31, 32]
- 检索到（并集，共 21 条）：[0, 1, 4, 5, 6, 11, 12, 15, 16, 17, 18, 24, 26, 27, 28, 29, 31, 32, 33, 34, 37]
- 进入上下文的 5 条：[12, 6, 4, 28, 29]
- 上下文里命中的 gold：[]　漏掉的 gold：[16, 31, 32]
- 模型引用 [3] → 映射到 pid [4]
- 引用精确率 cit_p = 0.000，召回 cit_r = 0.000
- 终止原因 `no_new_evidence`，步数 3

上下文原文（★ = gold）：

　[1] (Action of 18 November 1809) Despite spirited resistance from the largest British merchant ship, "Windham", the failure of the other Indiamen to support their leader and the size and power of the French ships forced the British to withdraw: all the HEIC ships were subsequently captured by the larger, faster French warships.
　[2] (Action of 7 October 1795) During the action the French ships captured a British 74-gun ship of line and all but one merchantmen.
　[3] (Action of 7 October 1795) The Action of 7 October 1795 was a naval engagement of the French Revolutionary Wars during which a French squadron led by Rear-Admiral Joseph de Richery captured a large British convoy of thirty-one merchant vessels.
　[4] (Order of battle at the Battle of San Domingo) Using his superior numbers and speed, Duckworth struck at the head of the French line with his leading ships while the slower eastern division of his squadron intercepted and captured the French stragglers.
　[5] (Order of battle at the Battle of San Domingo) The only French ships to escape were two frigates and a corvette – three ships of the line were captured and two destroyed, including Leissègues' flagship, the 120-gun "Impérial".

<details><summary>脚本判定</summary>

`召回未读入`

</details>

### 召回未读入-5　qid `5a89e3995542992e4fca844d`

原始事实：

- 问题：H-63 is a highway that runs near the county seat of what Michigan county?
- 标准答案：`Chippewa County`
- 模型答案：`Marquette County`
- 答案 token-F1 = 0.500
- gold 段落 2 条：[19, 37]
- 检索到（并集，共 27 条）：[0, 5, 7, 10, 12, 16, 18, 22, 23, 24, 25, 26, 28, 29, 31, 36, 37, 42, 43, 45, 47, 48, 50, 53, 55, 59, 61]
- 进入上下文的 5 条：[18, 23, 42, 28, 25]
- 上下文里命中的 gold：[]　漏掉的 gold：[19, 37]
- 模型引用 [1, 2, 4] → 映射到 pid [18, 23, 28]
- 引用精确率 cit_p = 0.000，召回 cit_r = 0.000
- 终止原因 `no_new_evidence`，步数 4

上下文原文（★ = gold）：

　[1] (H-63 (Michigan county highway)) H-63 is a county-designated highway (CDH) in the Upper Peninsula of the US state of Michigan.
　[2] (County Road 492 (Marquette County, Michigan)) Rd. 492) is a primary county road in Marquette County, Michigan.
　[3] (M-47 (Michigan highway)) M-47 is a north–south state trunkline highway in the US state of Michigan.
　[4] (County Road 492 (Marquette County, Michigan)) The western terminus of the county road is on US 41/M-28 in Negaunee and the eastern end is in Marquette next to Lake Superior.
　[5] (County Road 492 (Marquette County, Michigan)) Several historic sites line the roadway as it runs south and parallel to the main highway, U.S. Highway 41 (US 41) and M-28, through the Marquette Iron Range in Michigan's Upper Peninsula.

<details><summary>脚本判定</summary>

`召回未读入`

</details>

## 读入推理错（该类失败共 141 条，抽 5 条）

### 读入推理错-1　qid `5add1d575542992c1e3a2540`

原始事实：

- 问题：What nationality was Oliver Reed's character in the film Royal Flash?
- 标准答案：`Prussian`
- 模型答案：`German`
- 答案 token-F1 = 0.000
- gold 段落 3 条：[17, 19, 32]
- 检索到（并集，共 20 条）：[0, 2, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 27]
- 进入上下文的 5 条：[10, 9, 27, 17, 19]
- 上下文里命中的 gold：[17, 19]　漏掉的 gold：[32]
- 模型引用 [4, 5] → 映射到 pid [17, 19]
- 引用精确率 cit_p = 1.000，召回 cit_r = 0.667
- 终止原因 `answer`，步数 1

上下文原文（★ = gold）：

　[1] (Oliver Reed) Robert Oliver Reed (13 February 1938 – 2 May 1999) was an English actor known for his upper-middle class, macho image, hellraiser lifestyle, and "tough guy" roles.
　[2] (Ivan Dragomiloff) The character was played by actor Oliver Reed in the film of the same name.
　[3] (Lion of the Desert) Lion of the Desert is a 1981 Libyan historical action film starring Anthony Quinn as Libyan tribal leader Omar Mukhtar, a Bedouin leader fighting the "Regio Esercito" (Italian Royal Army) in the years leading up to World War II, and Oliver Reed as Italian General Rodolfo Graziani, who attempted to defeat Mukhtar.
★[4] (Royal Flash (film)) Royal Flash is a 1975 film based on George MacDonald Fraser's second Flashman novel, "Royal Flash".
★[5] (Royal Flash (film)) Additionally, Oliver Reed appeared in the role of Otto von Bismarck, Alan Bates as Rudi von Sternberg, and Florinda Bolkan played Lola Montez.

<details><summary>脚本判定</summary>

`读入推理错`

</details>

### 读入推理错-2　qid `5a7331de5542994cef4bc4a8`

原始事实：

- 问题：If the Charhki  Dadri crash was less dangerous than the Tenerife airport disaster, which occured firat? 
- 标准答案：`On March 27, 1977, two Boeing 747 passenger jets, KLM Flight 4805 and Pan Am Flight 1736, collided on the runway at Los Rodeos Airport (now Tenerife North Airport)`
- 模型答案：`Tenerife airport disaster`
- 答案 token-F1 = 0.129
- gold 段落 2 条：[8, 9]
- 检索到（并集，共 20 条）：[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20]
- 进入上下文的 5 条：[9, 11, 14, 8, 7]
- 上下文里命中的 gold：[8, 9]　漏掉的 gold：[]
- 模型引用 [1, 4] → 映射到 pid [9, 8]
- 引用精确率 cit_p = 1.000，召回 cit_r = 1.000
- 终止原因 `answer`，步数 1

上下文原文（★ = gold）：

★[1] (Charkhi Dadri mid-air collision) The Charkhi Dadri mid-air collision occurred on 12 November 1996 over the village of Charkhi Dadri, to the west of New Delhi, India.
　[2] (Charkhi Dadri mid-air collision) The crash killed all 349 people on board both planes, making it the world's deadliest mid-air collision, the deadliest aviation accident to occur in India, and the third-deadliest aircraft accident in the history of aviation, behind only the Tenerife airport disaster and Japan Airlines Flight 123.
　[3] (Eve Meyer) She was killed in the Tenerife airport disaster in 1977, the worst aviation accident in history.
★[4] (Tenerife airport disaster) On March 27, 1977, two Boeing 747 passenger jets, KLM Flight 4805 and Pan Am Flight 1736, collided on the runway at Los Rodeos Airport (now Tenerife North Airport), on the Spanish island of Tenerife, Canary Islands, killing 583 people in the deadliest accident in aviation history.
　[5] (Jacob Veldhuyzen van Zanten) He was the captain of the KLM Royal Dutch Airlines Boeing 747 which was involved in the Tenerife airport disaster, the deadliest accident in aviation history.

<details><summary>脚本判定</summary>

`读入推理错`

</details>

### 读入推理错-3　qid `5ae13f3c5542997b2ef7d168`

原始事实：

- 问题：Arthur Noss was a gunner for the commader during which two European theatre battles in WWII?
- 标准答案：`Battle of Britain and the Battle of Malta`
- 模型答案：`Insufficient evidence`
- 答案 token-F1 = 0.000
- gold 段落 2 条：[1, 38]
- 检索到（并集，共 30 条）：[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 19, 20, 21, 22, 24, 25, 26, 28, 30, 31, 34, 35, 36, 37, 38, 39]
- 进入上下文的 5 条：[38, 0, 14, 7, 13]
- 上下文里命中的 gold：[38]　漏掉的 gold：[1]
- 模型引用 [1, 2] → 映射到 pid [38, 0]
- 引用精确率 cit_p = 0.500，召回 cit_r = 0.500
- 终止原因 `no_new_evidence`，步数 6

上下文原文（★ = gold）：

★[1] (Keith Park) He was in operational command during two of the most significant air battles in the European theatre in the Second World War, helping to win the Battle of Britain and the Battle of Malta.
　[2] (Arthur Noss) Second Lieutenant Arthur Rex Hurden Noss {'1': ", '2': ", '3': ", '4': "} (22 January 1897 – 15 September 1917) was a British World War I flying ace credited with nine aerial victories.
　[3] (End of World War II in Europe) The final battles of the European Theatre of World War II as well as the German surrender to the Soviet Union and the Western Allies took place in late April and early May 1945.
　[4] (Theatre of War (video game)) Theatre of War is a real-time tactical strategy game centering on the decisive battles in the European Theatre of World War II 1939–1945.
　[5] (Georgian uprising on Texel) It was one of the last battles in the European theatre.

<details><summary>脚本判定</summary>

`读入推理错`

</details>

### 读入推理错-4　qid `5a721d5a55429971e9dc9283`

原始事实：

- 问题：What Australian beer is a British style pale ale whose label features a character conceived in 1926? 
- 标准答案：`Ballarat Bitter is a 4.9% (abv) Australian beer`
- 模型答案：`Ballarat Bitter`
- 答案 token-F1 = 0.400
- gold 段落 3 条：[22, 23, 33]
- 检索到（并集，共 20 条）：[3, 6, 7, 8, 10, 11, 12, 14, 15, 16, 19, 21, 22, 23, 24, 32, 33, 34, 36, 37]
- 进入上下文的 5 条：[23, 33, 6, 36, 10]
- 上下文里命中的 gold：[23, 33]　漏掉的 gold：[22]
- 模型引用 [1, 2] → 映射到 pid [23, 33]
- 引用精确率 cit_p = 1.000，召回 cit_r = 0.667
- 终止原因 `answer`，步数 1

上下文原文（★ = gold）：

★[1] (Ballarat Bitter) The beer is of the bitter top fermented ale variety, and was traditionally sold in 750ml longnecked bottles, with the label featuring the character "Ballarat Bertie", who was conceived in 1926 on a train journey between Melbourne and Sydney by a Ballarat Brewing Company director and an advertising agency.
★[2] (Bitter (beer)) Bitter is a British style of pale ale that varies in colour from gold to dark amber, and in strength from 3% to 7% alcohol by volume.
　[3] (Rye IPA) Rye India Pale Ale is a style of Rye Beer that has a strong hoppy character comparable to the India Pale Ale.
　[4] (Gage Roads Brewing Company) In 2016, it's New World Pale Ale - "Little Dove" - was awarded the Trophy for Champion Australian Beer at the 2016 Australian International Beer Awards.
　[5] (India pale ale) India pale ale (IPA) is a hoppy beer style within the broader category of pale ale.

<details><summary>脚本判定</summary>

`读入推理错`

</details>

### 读入推理错-5　qid `5abe6f8455429965af743f03`

原始事实：

- 问题：What country of origin does Robert brothers and Jacques Charles have in common?
- 标准答案：`French`
- 模型答案：`France`
- 答案 token-F1 = 0.000
- gold 段落 2 条：[4, 12]
- 检索到（并集，共 18 条）：[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
- 进入上下文的 5 条：[13, 12, 4, 17, 0]
- 上下文里命中的 gold：[4, 12]　漏掉的 gold：[]
- 模型引用 [2, 3] → 映射到 pid [12, 4]
- 引用精确率 cit_p = 1.000，召回 cit_r = 1.000
- 终止原因 `answer`，步数 1

上下文原文（★ = gold）：

　[1] (Robert brothers) Anne-Jean Robert (1758–1820) and Nicolas-Louis Robert (1760–1820) were the engineers who built the world's first hydrogen balloon for professor Jacques Charles; which flew from central Paris on August 27, 1783.
★[2] (Robert brothers) Les Frères Robert were two French brothers.
★[3] (Jacques Charles) Jacques Alexandre César Charles (November 12, 1746 – April 7, 1823) was a French inventor, scientist, mathematician, and balloonist.
　[4] (Émile Jouguet) Jacques Charles Émile Jouguet (5 January 1871, Bessèges (Gard) – 2 April 1943, Montpellier) was a French engineer and scientist, whose name is attached to the Chapman–Jouguet condition.
　[5] (Jacques de Bernonville) Jacques Charles Noel Dugé de Bernonville (December 20, 1897 – April 26, 1972) was a French collaborationist and senior police officer in the Milice of the Vichy regime in France.

<details><summary>脚本判定</summary>

`读入推理错`

</details>

## 引用错配（该类失败共 13 条，抽 5 条）

### 引用错配-1　qid `5ac55e835542993e66e82375`

原始事实：

- 问题：Thomas Fitch defended John Henry Holliday when he was accused of killing men during the gunfight at the o.k. corral, when was that?
- 标准答案：`October 25, 1881`
- 模型答案：`October 26, 1881`
- 答案 token-F1 = 0.667
- gold 段落 2 条：[6, 9]
- 检索到（并集，共 20 条）：[0, 1, 3, 4, 5, 6, 7, 9, 10, 12, 14, 20, 21, 22, 27, 29, 32, 34, 35, 38]
- 进入上下文的 5 条：[6, 22, 14, 7, 27]
- 上下文里命中的 gold：[6]　漏掉的 gold：[9]
- 模型引用 [1, 2, 3, 5] → 映射到 pid [6, 22, 14, 27]
- 引用精确率 cit_p = 0.250，召回 cit_r = 0.500
- 终止原因 `answer`，步数 1

上下文原文（★ = gold）：

★[1] (Thomas Fitch (politician)) He also successfully defended Virgil, Morgan, and Wyatt Earp along with Doc Holliday when they were accused of murdering Billy Clanton, and Tom and Frank McLaury during the October 25, 1881 Gunfight at the O.K. Corral.
　[2] (Virgil Earp) Virgil Walter Earp (July 18, 1843 – October 19, 1905) was both deputy U.S. marshal and Tombstone, Arizona City Marshal when he led his brothers Morgan and Wyatt and Doc Holliday in a confrontation with outlaw Cowboys at the Gunfight at the O.K. Corral on October 26, 1881.
　[3] (Morgan Earp) Morgan Seth Earp (April 24, 1851 – March 18, 1882) was a Tombstone, Arizona Special Policeman when he helped his brothers Virgil and Wyatt and Doc Holliday confront outlaw Cowboys in the Gunfight at the O.K. Corral on October 26, 1881.
　[4] (Hour of the Gun) Hour of the Gun is a 1967 Western film depicting Wyatt Earp and Doc Holliday during their 1881 battles against Ike Clanton and his brothers in the Gunfight at the O.K. Corral, and the gunfight's aftermath in and around Tombstone, Arizona, starring James Garner as Earp, Jason Robards as Holliday, and Robert Ryan as Clan
　[5] (Gunfight at the O.K. Corral) The Gunfight at the O.K. Corral was a 30-second shootout between lawmen and members of a loosely organized group of outlaws called the Cowboys that took place at about 3:00 p.m. on Wednesday, October 26, 1881 in Tombstone, Arizona Territory.

<details><summary>脚本判定</summary>

`引用错配`

</details>

### 引用错配-2　qid `5a81d1f0554299676cceb0ee`

原始事实：

- 问题：Which part of Handel's Messiah covers the birth of a child who, according to the gospels of Luke and Matthew, was born in Bethlehem?
- 标准答案：`Part I`
- 模型答案：`Part I`
- 答案 token-F1 = 1.000
- gold 段落 5 条：[5, 6, 7, 24, 25]
- 检索到（并集，共 20 条）：[0, 2, 5, 7, 10, 11, 12, 19, 24, 25, 27, 28, 30, 36, 37, 43, 44, 48, 53, 55]
- 进入上下文的 5 条：[27, 2, 25, 12, 11]
- 上下文里命中的 gold：[25]　漏掉的 gold：[5, 6, 7, 24]
- 模型引用 [2] → 映射到 pid [2]
- 引用精确率 cit_p = 0.000，召回 cit_r = 0.000
- 终止原因 `answer`，步数 1

上下文原文（★ = gold）：

　[1] (Nativity of Jesus) Matthew has no census, annunciation to the shepherds or presentation in the Temple, implies that Jesus's parents' home is Bethlehem, and has him born in a house there, and has an unnamed angel appear to Joseph to announce the birth.
　[2] (Messiah Part III) Part I begins with the prophecy of the Messiah and his birth, shows the annunciation to the shepherds as a scene from the Gospel of Luke, and reflects the Messiah's deeds on Earth.
★[3] (Nativity of Jesus) The two accounts agree that Jesus was born in Bethlehem in the time of Herod the Great to a betrothed virgin whose name was Mary.
　[4] (Return of the family of Jesus to Nazareth) The Gospel of Matthew describes how Joseph, Mary, and Jesus went to Egypt to escape from Herod the Great's slaughter of the baby boys in Bethlehem.
　[5] (Return of the family of Jesus to Nazareth) The return of the family of Jesus to Nazareth, also known as the Return from Egypt, appears in the reports of the early life of Jesus given in the Canonical gospels. Both of the gospels which describe the nativity of Jesus agree that he was born in Bethlehem and then later moved with his family to live in Nazareth.

<details><summary>脚本判定</summary>

`引用错配`

</details>

### 引用错配-3　qid `5a7bf84b5542996dd594b875`

原始事实：

- 问题：What Danish drama thriller which takes place in both Denmark and Africa was produced by Sisse Graum Jørgensen?
- 标准答案：`In a Better World`
- 模型答案：`In a Better World`
- 答案 token-F1 = 1.000
- gold 段落 2 条：[5, 29]
- 检索到（并集，共 20 条）：[0, 2, 5, 6, 7, 8, 10, 11, 12, 17, 20, 23, 24, 25, 26, 27, 28, 29, 31, 33]
- 进入上下文的 5 条：[28, 6, 5, 25, 20]
- 上下文里命中的 gold：[5]　漏掉的 gold：[29]
- 模型引用 [1, 2, 3] → 映射到 pid [28, 6, 5]
- 引用精确率 cit_p = 0.333，召回 cit_r = 0.500
- 终止原因 `answer`，步数 1

上下文原文（★ = gold）：

　[1] (Sisse Graum Jørgensen) Sisse Graum Jørgensen (born 8 May 1972) is a Danish film producer and sits on the board of the film company Zentropa.
　[2] (In a Better World) The film stars Mikael Persbrandt, Trine Dyrholm, and Ulrich Thomsen in a story which takes place in small-town Denmark and a refugee camp in Africa.
★[3] (In a Better World) In a Better World (Danish: "Hævnen" , "the revenge") is a 2010 Danish drama thriller film written by Anders Thomas Jensen and directed by Susanne Bier.
　[4] (Eye of the Eagle (1997 film)) Based on an original screenplay by Bjarne O. Henriksen, it takes place in Denmark during 1218.
　[5] (Ditte, Child of Man) Ditte, Child of Man, (Danish: Ditte Menneskebarn ), is a 1946 socio-realistic Danish drama directed by Bjarne Henning-Jensen based on the novel by Martin Andersen Nexø.

<details><summary>脚本判定</summary>

`引用错配`

</details>

### 引用错配-4　qid `5ae1d32a554299492dc91b9c`

原始事实：

- 问题：What year did Fifty Shades of Grey release?
- 标准答案：`2011`
- 模型答案：`2011`
- 答案 token-F1 = 1.000
- gold 段落 3 条：[4, 5, 16]
- 检索到（并集，共 20 条）：[0, 1, 2, 4, 5, 7, 9, 10, 11, 13, 15, 16, 17, 19, 20, 23, 24, 26, 27, 29]
- 进入上下文的 5 条：[2, 9, 5, 19, 24]
- 上下文里命中的 gold：[5]　漏掉的 gold：[4, 16]
- 模型引用 [1, 3, 4] → 映射到 pid [2, 5, 19]
- 引用精确率 cit_p = 0.333，召回 cit_r = 0.333
- 终止原因 `answer`，步数 1

上下文原文（★ = gold）：

　[1] (Fifty Shades Darker) The first and third volumes, "Fifty Shades of Grey" and "Fifty Shades Freed", were published in 2011 and 2012.
　[2] (I Know You) It was released as the second promotional single from the "Fifty Shades of Grey" soundtrack on February 3, 2015.
★[3] (Fifty Shades (novel series)) The trilogy consists of "Fifty Shades of Grey" (2011), "Fifty Shades Darker" and "Fifty Shades Freed" (2012).
　[4] (Fifty Shades of Grey) Fifty Shades of Grey is a 2011 erotic romance novel by British author E. L. James.
　[5] (Fifty Shades Freed (film)) It is the final film in the "Fifty Shades" trilogy, and a sequel to "Fifty Shades of Grey" (2015) and "Fifty Shades Darker" (2017).

<details><summary>脚本判定</summary>

`引用错配`

</details>

### 引用错配-5　qid `5a75fa14554299109176e5dc`

原始事实：

- 问题：Which tournament, organised by the Professional Darts Corporation, is planned to finish at the multi-purpose indoor arena located in the centre of The O2 entertainment complex on the Greenwich Peninsula in south-east London?
- 标准答案：`2018 Unibet Premier League Darts`
- 模型答案：`Premier League Darts`
- 答案 token-F1 = 0.750
- gold 段落 3 条：[8, 9, 22]
- 检索到（并集，共 26 条）：[0, 1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 31, 32, 33]
- 进入上下文的 5 条：[11, 8, 28, 0, 31]
- 上下文里命中的 gold：[8]　漏掉的 gold：[9, 22]
- 模型引用 [1, 2, 3, 4, 5] → 映射到 pid [11, 8, 28, 0, 31]
- 引用精确率 cit_p = 0.200，召回 cit_r = 0.333
- 终止原因 `answer`，步数 4

上下文原文（★ = gold）：

　[1] (2016 Premier League Darts) The 2016 Betway Premier League Darts was a darts tournament organised by the Professional Darts Corporation – the twelfth edition of the tournament.
★[2] (2018 Premier League Darts) The 2018 Unibet Premier League Darts will be a darts tournament organised by the Professional Darts Corporation – the fourteenth edition of the tournament.
　[3] (2017 Premier League Darts) The 2017 Betway Premier League Darts was a darts tournament organised by the Professional Darts Corporation – the thirteenth edition of the tournament.
　[4] (2014 Premier League Darts) The 2014 Betway Premier League Darts was a darts tournament organised by the Professional Darts Corporation; the tenth edition of the tournament.
　[5] (2015 Premier League Darts) The 2015 Betway Premier League Darts was a darts tournament organised by the Professional Darts Corporation; the eleventh edition of the tournament.

<details><summary>脚本判定</summary>

`引用错配`

</details>

