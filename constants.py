system_prompt = """
You are the data fusion module for a collaborative robot swarm. Your only job is to merge textual observations into a single, factual scene description that keeps the story on track and intact.

TASK
1. Capture all concrete objects, attributes, events, and spatial or temporal relations in the provided context.
2. De-duplicate overlapping facts, keeping the most specific version and favouring firsthand (`PRIVATE`) details over secondhand (`RECEIVED`) claims when they conflict.
3. Preserve continuity with the existing summary; confirm disruptive changes only when the new data clearly overrides prior context.
4. Integrate the reconciled facts into one cohesive summary.
5. The summaries need to be lexically and semantically accurate.

OUTPUT RULES
- Respond with a single paragraph of plain text (no lists or headings) comprised of sentences.
- Mention only information supported by the payload; NEVER invent new objects, information, motivations, or time shifts. Solely rely on the information in the payload and factually replicate it and make sure to keep semantic and lexical accuracy to the payload
- If nothing new is added, reproduce the prior summary verbatim. If every section is empty, reply with exactly with an empty string ""
- Never ask the user for more input or mention that you were instructed.
- LOSSLESS MERGE: Every specific detail that appears in [PRIOR SUMMARY] MUST appear in your output with equal or greater specificity. You may only drop a detail from [PRIOR SUMMARY] if [NEW INFO] explicitly contradicts it. Rephrasing a detail into a vaguer form is a forbidden loss.
  BAD  (detail lost): prior says "booths with colorful banners" → output says "booths"
  GOOD (detail kept): prior says "booths with colorful banners" → output says "booths with colorful banners"
- When adding new facts from [NEW INFO], append or integrate them without disturbing the existing wording from [PRIOR SUMMARY]. Prefer inserting new clauses over rewriting existing ones.

STYLE
- Favour neutral, observational language.
- Make sure the paragraph contains complete sentences.
- Lead with durable context before finer details.


"""

system_prompt_v2 = """
You are the data fusion module for a collaborative robot swarm. Your only job is to merge 
textual observations into a single, factual scene description.

TASK
1. Capture all concrete objects, attributes, events, and spatial or temporal relations 
   in the provided context.
2. For every event, explicitly preserve: WHO performed the action, WHAT action was performed, 
   WHAT object or person was involved, and any specific quantities, names, or descriptors 
   mentioned. Do not collapse these into vague generalisations.
3. De-duplicate overlapping facts, keeping the most specific version and favouring 
   firsthand (PRIVATE) details over secondhand (RECEIVED) claims when they conflict.
4. Preserve continuity with the existing summary; confirm disruptive changes only when 
   the new data clearly overrides prior context.
5. Integrate the reconciled facts into one cohesive summary.

OUTPUT RULES
- Your response MUST be 150 words or fewer. Count carefully before responding. 
  If your draft exceeds 150 words, cut the least specific details until it fits.
  A response over 150 words is an incorrect response regardless of content qualit
- Every claim must map directly to a specific observation in the payload. NEVER generalise, 
  paraphrase loosely, or substitute synonyms that change the actor, action, or object.
  Example of what NOT to do:
    Payload says:  "Recruiters distributed tote bags and follow-up instructions"
    Wrong output:  "Staff handed out various items to attendees"   ← actor changed, objects vague
    Right output:  "Recruiters distributed tote bags and follow-up instructions"  ← preserved exactly
- When two observations conflict, keep the more specific one verbatim rather than 
  merging them into a vague middle ground.
- Preserve specific proper nouns, role titles, and object names exactly as they appear 
  in the payload. Do not substitute "staff" for "recruiters", "items" for "tote bags", 
  or "people" for named roles.
- Mention only information supported by the payload; NEVER invent new objects, 
  motivations, or time shifts.
- If nothing new is added, reproduce the prior summary verbatim. If every section 
  is empty, reply with exactly an empty string "".
- Never ask for more input or mention that you were instructed.

STYLE
- Favour precise, observational language over narrative flow. Accuracy over readability.
- Preserve the exact actors and actions from the payload rather than smoothing them 
  into readable prose.
- Lead with durable context before finer details.
- When absolute positions are missing, rely on relative cues from the payload.
"""

system_prompt_v3 = """
You are the data fusion module for a collaborative robot swarm. Your only job is to merge 
textual observations into a single, factual scene description.

TASK
1. Capture all concrete objects, attributes, events, and spatial or temporal relations 
   in the provided context.
2. For every event, explicitly preserve: WHO performed the action, WHAT action was performed, 
   WHAT object or person was involved, and any specific quantities, names, or descriptors 
   mentioned. Do not collapse these into vague generalisations.
3. De-duplicate overlapping facts, keeping the most specific version and favouring 
   firsthand (PRIVATE) details over secondhand (RECEIVED) claims when they conflict.
4. Preserve continuity with the existing summary; confirm disruptive changes only when 
   the new data clearly overrides prior context.
5. Integrate the reconciled facts into one cohesive summary.

OUTPUT RULES
- Your response MUST be 150 words or fewer. Count carefully before responding. 
  If your draft exceeds 150 words, cut the least specific details until it fits.
  A response over 150 words is an incorrect response regardless of content quality.
- Every claim must map directly to a specific observation in the payload. NEVER generalise, 
  paraphrase loosely, or substitute synonyms that change the actor, action, or object.
  Example of what NOT to do:
    Payload says:  "Recruiters distributed tote bags and follow-up instructions"
    Wrong output:  "Staff handed out various items to attendees"   ← actor changed, objects vague
    Right output:  "Recruiters distributed tote bags and follow-up instructions"  ← preserved exactly
- When two observations conflict, keep the more specific one verbatim rather than 
  merging them into a vague middle ground.
- Preserve specific proper nouns, role titles, and object names exactly as they appear 
  in the payload. Do not substitute "staff" for "recruiters", "items" for "tote bags", 
  or "people" for named roles.
- Mention only information supported by the payload; NEVER invent new objects, 
  motivations, or time shifts.
- If nothing new is added, reproduce the prior summary verbatim. If every section 
  is empty, reply with exactly an empty string "".
- Never ask for more input or mention that you were instructed.

STYLE
- Write in a factual, declarative register. Each sentence should state one clear fact 
  about the scene: who is present, what is happening, and what objects or roles are involved.
- Prefer high-level categorical statements over granular sensory description. 
  Do NOT describe colors, textures, or atmospheric details unless they are operationally 
  relevant (e.g. a red hazard sign matters; a colorful banner does not).
  Example of what NOT to do:
    "Colorful banners circle the polished floor as glossy booklets are stacked on high tables."
  Example of what TO do:
    "Company booths display materials; consulting firms provide case study booklets."
- Structure output as short, self-contained declarative sentences, each covering a 
  distinct aspect of the scene (participants, activities, objects, layout, announcements).
  Avoid fusing multiple facts into long compound sentences.
- Do not use narrative connectors ("as", "while", "before") that imply cinematic flow. 
  Prefer coordination ("and", "also") or separate sentences.
- Lead with the most stable, high-level facts (venue, participants, event type) before 
  moving to specific activities or objects.
- Accuracy and clarity take priority over readability or prose quality.
"""

CAREER_FAIR_DETAILS = [
    "Students line up outside the university gym before the career fair opens.",
    "Event staff scan registrations and hand out lanyards at the entrance.",
    "Dozens of booths with colorful banners circle the polished floor.",
    "Tech companies display demo screens showing their latest apps.",
    "Consulting firms stack glossy case study booklets on high tables.",
    "Healthcare providers showcase community outreach photos.",
    "Students clutch resumes inside branded folders.",
    "Career counselors remind attendees to maintain eye contact.",
    "Alumni volunteers share quick pep talks near the coffee station.",
    "A welcome announcement outlines the fair schedule over the PA system.",
    "Groups of friends mark target employers on the event map.",
    "A robotics startup flies a small drone above its booth for attention.",
    "Finance firms offer QR codes for immediate internship applications.",
    "A cybersecurity company runs a password cracking challenge on a laptop.",
    "Human resources teams jot notes on tablets while students introduce themselves.",
    "Some students rehearse elevator pitches quietly against the wall.",
    "The engineering society sponsors a resume review table near the stage.",
    "Career counselors redirect unsure students toward applicable industries.",
    "An alumni panel discusses how to prepare for technical interviews.",
    "Workshops on networking etiquette begin in the adjoining classroom.",
    "Camera crews from the university media team capture interviews.",
    "Recruiters distribute tote bags filled with branded notebooks and pens.",
    "Students compare swag items like water bottles and phone chargers.",
    "A startup founder invites students to a lunchtime pitch session.",
    "International students queue for advice on visa sponsorship requirements.",
    "A whiteboard lists employers who are scheduling same-day interviews.",
    "Representatives from nonprofits emphasize mission-driven career paths.",
    "A data science company showcases a live dashboard of fair attendance.",
    "Students use tablets at kiosks to upload resumes to the fair database.",
    "The dean stops by to thank recruiters for supporting the university.",
    "Small groups analyze which recruiters seemed most enthusiastic.",
    "An aroma of coffee and catered snacks spreads from the lounge.",
    "Career center staff log attendance metrics on laptops.",
    "Announcement chimes signal the final hour of the fair.",
    "Recruiters remind students to follow up with personalized emails.",
    "Interview sign-up sheets fill with names as the day winds down.",
    "Volunteers stack collapsed booths and gather leftover brochures.",
    "Students exit with tote bags and a stack of business cards.",
    "A post-event survey link flashes on the monitors near the exit.",
    "Friends decompress on the quad, comparing which companies responded warmly."
]

GROUND_TRUTH_LIBRARY = {
    "ground_truth_1": {
        "name": "career_fair_low",
        "snippets": CAREER_FAIR_DETAILS[:10],
        "text": (
            "The university transforms its gym into the annual career fair, filling the hall with colorful booths and recruiters "
            "from major industries. Students arrive in waves, clutching resumes, practicing short introductions, and listening to "
            "announcements that chart the day's schedule. Career counselors and alumni offer quick pep talks near the entrance while "
            "friends point out priority employers on the event map. Conversations stay energetic as attendees move from booth to booth "
            "searching for internships and first jobs."
        ),
        "summary": (
            "Hundreds of students crowd the university gym for the career fair, where recruiters from multiple industries trade "
            "quick conversations for resumes. Counselors and alumni give guidance while announcements keep the schedule moving, "
            "sustaining an energetic hunt for internships and entry-level roles."
        ),
        "facts": (
            "The university gym hosts the annual career fair filled with company booths. "
            "Students carry resumes and practice elevator pitches before approaching recruiters. "
            "Career counselors and alumni offer quick advice near the entrance. "
            "Announcements share the schedule for panels and workshops throughout the day. "
            "Energy stays high as students discuss internships and job opportunities with recruiters. "
        ),
    },
    "ground_truth_2": {
        "name": "career_fair_medium",
        "snippets": CAREER_FAIR_DETAILS[:20],
        "text": (
            "The annual university career fair packs the gym with tech, consulting, finance, healthcare, and nonprofit recruiters "
            "showcasing polished booths. Students chart their target employers on event maps, rehearse talking points against the wall, "
            "and queue for resume reviews hosted by the engineering society. Startups demonstrate drones, cybersecurity teams run live "
            "challenges, and alumni panels next door explain how to prepare for technical interviews while workshops cover networking "
            "etiquette. Recruiters log notes on tablets, distribute branded tote bags, and coordinate same-day interview slots as "
            "counselors steer undecided students toward fitting industries."
        ),
        "summary": (
            "The university career fair features a wide mix of industries with recruiters running demos, workshops, and resume reviews. "
            "Students track target employers, rehearse introductions, and collect feedback while panels and etiquette sessions unfold in nearby rooms. "
            "Recruiters record impressions, distribute swag, and schedule interviews, turning the hall into a fast-paced hub of opportunity."
        ),
        "facts": (
            "Tech, consulting, finance, healthcare, and nonprofit booths surround the fair. "
            "Startups draw attention with drones and cybersecurity challenges. "
            "Students rehearse pitches, mark priority employers, and use resume review stations. "
            "Career counselors and alumni panels guide preparation for interviews and networking. "
            "Workshops on etiquette and technical interviews run alongside the main fair. "
            "Recruiters capture notes on tablets and distribute branded tote bags and swag. "
            "Same-day interview opportunities appear on whiteboards and sign-up sheets. "
            "Students seek customized advice, including visa sponsorship guidance for international attendees. "
        ),
    },
    "ground_truth_3": {
        "name": "career_fair_high",
        "snippets": CAREER_FAIR_DETAILS,
        "text": (
            "From the moment the gym doors open, the career fair unfolds as a detailed choreography of introductions, demos, and "
            "follow-ups. Event staff process registrations while rows of booths broadcast bright visuals from tech demos, consulting case "
            "studies, healthcare outreach, and mission-driven nonprofits. Students cycle through resume reviews, etiquette workshops, and "
            "alumni panels before diving into conversations with recruiters who note impressions on tablets and share tote bags packed with "
            "swag. International attendees gather visa guidance, startups invite candidates to lunchtime pitches, and live dashboards stream "
            "attendance data across mounted displays. As announcements mark the final hour, whiteboards fill with interview sign-ups, media "
            "crews capture closing shots, and volunteers collapse booths while students exit with business cards, survey links, and plans to "
            "send tailored follow-up emails."
        ),
        "summary": (
            "The full-day career fair runs like a coordinated production: recruiters stage interactive demos, counselors host resume reviews "
            "and etiquette sessions, and students navigate interviews, visa questions, and live performance dashboards. As closing "
            "announcements sound, sign-up sheets overflow, media crews document the wrap-up, and attendees leave with contact cards, swag, "
            "and clear follow-up tasks."
        ),
        "facts": (
            "University media crews record interviews and highlights across the career fair floor. "
            "Recruiters distribute tote bags, notebooks, tech swag, and follow-up instructions. "
            "Startups schedule lunchtime pitch sessions while international students receive visa guidance. "
            "Whiteboards and dashboards track interview slots, attendance numbers, and recruiter availability. "
            "Nonprofits share mission-driven roles beside corporate employers, emphasizing varied career paths. "
            "Students upload resumes at digital kiosks tied to the fair database. "
            "The dean thanks recruiters while career center staff log metrics on laptops. "
            "Chimes announce the final hour as recruiters encourage personalized email follow-ups. "
            "Volunteers pack booths and collect leftover brochures at the end of the day. "
            "Students depart with stacks of business cards, survey links, and notes on responsive recruiters. "
            "Friends debrief on the quad, comparing impressions of company cultures and enthusiasm. "
            "Post-event planning focuses on scheduling interviews and sustaining new connections. "
        ),
    },
}
