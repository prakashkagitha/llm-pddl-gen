(define (problem coin_collector_numLocations7_numDistractorItems0_seed86)
  (:domain coin-collector)
  (:objects
    kitchen pantry living_room corridor bathroom backyard bedroom - room
    north south east west - direction
    coin - item
  )
  (:init
    (at kitchen)
    (closed-door kitchen pantry north)
    (connected kitchen living_room south)
    (closed-door pantry kitchen south)
    (connected living_room kitchen north)
    (connected living_room corridor south)
    (closed-door living_room bathroom east)
    (closed-door living_room backyard west)
    (connected corridor living_room north)
    (closed-door corridor bedroom east)
    (closed-door bathroom living_room west)
    (connected bathroom bedroom south)
    (closed-door backyard living_room east)
    (closed-door bedroom corridor west)
    (connected bedroom bathroom north)
    (location coin bedroom)
    (is-reverse north south)
    (is-reverse south north)
    (is-reverse east west)
    (is-reverse west east)
  )
  (:goal 
    (taken coin)
  )
)