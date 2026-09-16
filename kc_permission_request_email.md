To: giscenter@kingcounty.gov
Subject: Permission request - non-commercial use of eRealProperty photos for a Seattle architectural history project

Hello,

I'm working on a non-commercial, educational web project that maps and
classifies the architectural style of single-family homes in Seattle (e.g.
Craftsman, Tudor Revival, Mid-Century Modern, etc.), and I'd like to request
permission to use property photos from the Assessor's eRealProperty system
(blue.kingcounty.com/Assessor/eRealProperty) as part of it.

What I'd like to do:
- For Seattle's ~128,000 single-family residential parcels (identified via
  the PARCEL_ADDRESS_PUB_AREA public dataset on King County's GIS Open Data
  site, which I'm already using under its open terms), retrieve the
  Assessor's exterior property photo for each parcel.
- Use each photo, one at a time, as input to an image classification step
  that estimates the home's architectural style.
- Display the photo, address, and estimated style together on a public,
  non-commercial web map.

Why I'm asking first: I reviewed the King County website Terms of Use, which
state that content may not be published, displayed, distributed, or
commercially exploited without prior written permission. I want to make sure
this specific use is acceptable before proceeding, since it involves
systematically retrieving and publicly displaying photos rather than
one-off individual lookups.

A few things I want to be upfront about:
- This project is non-commercial and educational/civic in nature - no ads,
  no sale of data, no paywall.
- I would fetch photos gradually and respectfully (throttled to roughly one
  request per second, processed one Seattle neighborhood at a time, spread
  out over about two months) rather than all at once, to keep load on your
  systems minimal.
- I'm glad to credit King County Department of Assessments as the photo
  source on the site, link back to the eRealProperty system, and/or follow
  any other conditions (e.g., a specific attribution format, excluding
  certain parcels, a size/resolution limit) you'd want to set.
- I'm also happy to share the project's scope, code, or a working
  prototype if that's helpful for your review.

Could you let me know whether this use would be permitted, and if so,
under what conditions? Happy to answer any questions or adjust the
approach as needed.

Thank you for your time,
[Your name]
[Your email / contact info]
